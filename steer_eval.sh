#!/bin/bash
# Interactive steering eval — run this inside an srun --pty session.
#
# Usage:
#   srun --pty --account=weirdlab --partition=ckpt-all \
#        --gres=gpu:a40:1 --cpus-per-task=16 --mem=64G bash
#   # then inside that shell:
#   bash steer_eval.sh [options]
#
# Options:
#   --steer-frequency N    Prompt every N steps (default: 50)
#   --rollouts N           Number of episodes (default: 1)
#   --environment ENV      IsaacLab env name (default: WIDOWX-FoodBussing)
#   --policy-client NAME   Policy client name (default: WidowXJointPos)
#   --port PORT            VLA server port (default: 8001)
#   --instruction TEXT     Starting language instruction override

set -e

# ── Defaults ───────────────────────────────────────────────────────────────────
STEER_FREQUENCY=50
ROLLOUTS=1
EVAL_ENV="${EVAL_ENV:-WIDOWX-FoodBussing}"
POLICY_CLIENT="${POLICY_CLIENT:-WidowXJointPos}"
POLICY_PORT="${POLICY_PORT:-8001}"
INSTRUCTION=""

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steer-frequency) STEER_FREQUENCY="$2"; shift 2 ;;
        --rollouts)        ROLLOUTS="$2";         shift 2 ;;
        --environment)     EVAL_ENV="$2";         shift 2 ;;
        --policy-client)   POLICY_CLIENT="$2";    shift 2 ;;
        --port)            POLICY_PORT="$2";      shift 2 ;;
        --instruction)     INSTRUCTION="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Paths ──────────────────────────────────────────────────────────────────────
UV=/mmfs1/gscratch/weirdlab/dg20/UV/bin/uv
UV_CACHE_DIR=/mmfs1/gscratch/weirdlab/dg20/UV/cache
POLARIS_DIR=/mmfs1/gscratch/weirdlab/dg20/polaris
ISAAC_SIF=/mmfs1/gscratch/weirdlab/dg20/Isaac/octi-lab-base.sif
SVLA_REPO=/mmfs1/gscratch/weirdlab/dg20/steerable-policies-maniskill
SVLA_SIF=${SVLA_REPO}/maniskill_vla.sif
SVLA_CHECKPOINT="${SVLA_CHECKPOINT:-${SVLA_REPO}/.cache/huggingface/hub/models--Embodied-CoT--steerable-policy-openvla-7b-bridge/snapshots/adfc02fe284d0c643a4f1fdb895db907e31a0d61/checkpoints/step-080000-epoch-09-loss=0.0506.pt}"
SVLA_UNNORM_KEY="${SVLA_UNNORM_KEY:-bridge_orig}"
VENV=${POLARIS_DIR}/.venv
CUDA_HOME_SYNTH=${POLARIS_DIR}/.cuda_home
RUN_FOLDER="${RUN_FOLDER:-${POLARIS_DIR}/runs/steer_$(date +%Y%m%d_%H%M%S)}"

# Build extra eval args
EXTRA_EVAL_ARGS=""
if [[ -n "$INSTRUCTION" ]]; then
    EXTRA_EVAL_ARGS="--instruction '$INSTRUCTION'"
fi

echo "=== PolaRiS Interactive Steering Eval ==="
echo "Node:             $(hostname)"
echo "Date:             $(date)"
echo "Environment:      ${EVAL_ENV}"
echo "Policy client:    ${POLICY_CLIENT}"
echo "Steer frequency:  every ${STEER_FREQUENCY} steps"
echo "Rollouts:         ${ROLLOUTS}"
echo "Port:             ${POLICY_PORT}"
echo "Run folder:       ${RUN_FOLDER}"
[[ -n "$INSTRUCTION" ]] && echo "Instruction:      ${INSTRUCTION}"
mkdir -p "${RUN_FOLDER}"

# Load HF_TOKEN
# shellcheck source=/dev/null
set -a; source "${SVLA_REPO}/.env"; set +a

# ── Cleanup ────────────────────────────────────────────────────────────────────
SERVER_PID=""
cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    if [[ -n "${SERVER_PID}" ]]; then
        echo "Killing VLA server (PID ${SERVER_PID})"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    echo "Results saved to: ${RUN_FOLDER}"
}
trap cleanup EXIT

# ── Step 1: Start VLA server ───────────────────────────────────────────────────
SERVER_LOG="${RUN_FOLDER}/server.log"
echo ""
echo "=== Starting VLA server on port ${POLICY_PORT} ==="
echo "    Log: ${SERVER_LOG}"

apptainer exec \
    --nv --bind /mmfs1 \
    --env "HF_TOKEN=${HF_TOKEN}" \
    --env "HF_HOME=${SVLA_REPO}/.cache/huggingface" \
    --env "TRANSFORMERS_CACHE=${SVLA_REPO}/.cache/huggingface/hub" \
    --env "TORCH_HOME=${SVLA_REPO}/.cache/torch" \
    "${SVLA_SIF}" \
    bash -c "
        pip install flask --quiet --disable-pip-version-check 2>&1
        python -u ${POLARIS_DIR}/scripts/serve_steerable_vla.py \
            --checkpoint '${SVLA_CHECKPOINT}' \
            --repo '${SVLA_REPO}' \
            --unnorm-key '${SVLA_UNNORM_KEY}' \
            --port ${POLICY_PORT} \
            --trace-log '${RUN_FOLDER}/svla_trace.log'
    " > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# ── Step 2: Wait for server ────────────────────────────────────────────────────
echo ""
echo "=== Waiting for server (model load + warmup, ~2-3 min) ==="
MAX_WAIT=1800
ELAPSED=0
while ! nc -z localhost "${POLICY_PORT}" 2>/dev/null; do
    if [[ "${ELAPSED}" -ge "${MAX_WAIT}" ]]; then
        echo "ERROR: Server did not start within ${MAX_WAIT}s"
        exit 1
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: Server process died"
        tail -20 "${SERVER_LOG}" 2>/dev/null
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [[ $((ELAPSED % 60)) -eq 0 ]]; then
        echo "  ${ELAPSED}s | $(tail -1 "${SERVER_LOG}" 2>/dev/null)"
    else
        echo "  ${ELAPSED}s..."
    fi
done
echo "Server ready after ${ELAPSED}s"

# ── Step 3: Run evaluator (interactive, stdin forwarded) ───────────────────────
echo ""
echo "=== Running evaluator (steer every ${STEER_FREQUENCY} steps) ==="
echo "    At each prompt: type a new instruction, or press Enter to keep current."
echo ""

apptainer exec \
    --nv --bind /mmfs1 \
    --env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
    --env "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" \
    --env "OMNI_KIT_ACCEPT_EULA=Y" \
    --env "CUDA_HOME=${CUDA_HOME_SYNTH}" \
    --env "TORCH_CUDA_ARCH_LIST=8.6+PTX" \
    --env "LD_LIBRARY_PATH=${VENV}/lib/python3.11/site-packages/nvidia/cu13/lib" \
    --env "NO_PROXY=localhost,127.0.0.1,::1" \
    --env "no_proxy=localhost,127.0.0.1,::1" \
    "${ISAAC_SIF}" \
    bash -c "
        cd ${POLARIS_DIR}
        ${UV} run scripts/eval.py \
            --environment ${EVAL_ENV} \
            --policy.client ${POLICY_CLIENT} \
            --policy.host 127.0.0.1 \
            --policy.port ${POLICY_PORT} \
            --run-folder ${RUN_FOLDER} \
            --rollouts ${ROLLOUTS} \
            --steer-frequency ${STEER_FREQUENCY} \
            ${EXTRA_EVAL_ARGS}
    "

echo ""
echo "=== Done. Results in: ${RUN_FOLDER} ==="
