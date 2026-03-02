#!/bin/bash
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt-all
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/mmfs1/gscratch/weirdlab/dg20/slurm_logs/eval_svla_%j.out
#SBATCH --error=/mmfs1/gscratch/weirdlab/dg20/slurm_logs/eval_svla_%j.err
#SBATCH --job-name=polaris_svla

# Usage:
#   sbatch eval_steerable_vla.sh
#
# Override defaults via --export, e.g.:
#   sbatch --export=ALL,EVAL_ENV=DROID-FoodBussing eval_steerable_vla.sh
#   sbatch --export=ALL,EVAL_ENV=WIDOWX-FoodBussing eval_steerable_vla.sh

set -e
echo "=== PolaRiS Steerable-VLA Evaluation ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
nvidia-smi | head -10

# ── Paths ──────────────────────────────────────────────────────────────────────
UV=/mmfs1/gscratch/weirdlab/dg20/UV/bin/uv
UV_CACHE_DIR=/mmfs1/gscratch/weirdlab/dg20/UV/cache
POLARIS_DIR=/mmfs1/gscratch/weirdlab/dg20/polaris
ISAAC_SIF=/mmfs1/gscratch/weirdlab/dg20/Isaac/octi-lab-base.sif

SVLA_REPO=/mmfs1/gscratch/weirdlab/dg20/steerable-policies-maniskill
SVLA_SIF=${SVLA_REPO}/maniskill_vla.sif
SVLA_CHECKPOINT="${SVLA_CHECKPOINT:-${SVLA_REPO}/.cache/huggingface/hub/models--Embodied-CoT--steerable-policy-openvla-7b-bridge/snapshots/adfc02fe284d0c643a4f1fdb895db907e31a0d61/checkpoints/step-080000-epoch-09-loss=0.0506.pt}"
SVLA_UNNORM_KEY="${SVLA_UNNORM_KEY:-bridge_orig}"

# Load HF_TOKEN from steerable-policies .env (needed for gated Llama-2 backbone)
# shellcheck source=/dev/null
set -a; source "${SVLA_REPO}/.env"; set +a

# ── Eval settings ──────────────────────────────────────────────────────────────
POLICY_PORT="${POLICY_PORT:-8001}"
EVAL_ENV="${EVAL_ENV:-WIDOWX-FoodBussing}"
POLICY_CLIENT="${POLICY_CLIENT:-WidowXJointPos}"
RUN_FOLDER="${RUN_FOLDER:-${POLARIS_DIR}/runs/eval_svla_$(date +%Y%m%d_%H%M%S)}"
EVAL_ARGS="${EVAL_ARGS:-}"

echo "Checkpoint:   ${SVLA_CHECKPOINT}"
echo "Unnorm key:   ${SVLA_UNNORM_KEY}"
echo "Server port:  ${POLICY_PORT}"
echo "Environment:  ${EVAL_ENV}"
echo "Policy client: ${POLICY_CLIENT}"
echo "Run folder:   ${RUN_FOLDER}"
mkdir -p "${RUN_FOLDER}"

# ── Cleanup ────────────────────────────────────────────────────────────────────
cleanup() {
    echo "=== Cleaning up ==="
    if [ -n "${SERVER_PID}" ]; then
        echo "Killing steerable-VLA server (PID ${SERVER_PID})"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    if [ -f "${SERVER_LOG:-}" ]; then
        echo "=== Last 40 lines of server log ==="
        tail -40 "${SERVER_LOG}"
    fi
    if [ -f "${RUN_FOLDER}/svla_trace.log" ]; then
        echo "=== Trace log ==="
        cat "${RUN_FOLDER}/svla_trace.log"
    fi
    echo "Done."
}
trap cleanup EXIT

# ── Step 1: Start the steerable-VLA server (maniskill_vla.sif) ────────────────
# fastapi/uvicorn are not in maniskill_vla.sif so we pip-install them at runtime
SERVER_LOG="${RUN_FOLDER}/server.log"
echo "=== Starting steerable-VLA server on port ${POLICY_PORT} ==="
echo "    Server log: ${SERVER_LOG}"
apptainer exec \
    --nv \
    --bind /mmfs1 \
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
echo "Steerable-VLA server started with PID ${SERVER_PID}"

# ── Step 2: Wait for server to be ready ───────────────────────────────────────
# Port only opens after load + warmup inference complete (see serve_steerable_vla.py)
echo "=== Waiting for server on port ${POLICY_PORT} ==="
MAX_WAIT=1800   # 30 min – model load + warmup
ELAPSED=0
while ! nc -z localhost "${POLICY_PORT}" 2>/dev/null; do
    if [ "${ELAPSED}" -ge "${MAX_WAIT}" ]; then
        echo "ERROR: Server did not start within ${MAX_WAIT}s"
        exit 1
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: Server process died (PID ${SERVER_PID})"
        echo "=== Server log tail ==="
        tail -20 "${SERVER_LOG}" 2>/dev/null || echo "(no server log)"
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    # Every 60s, show last server log line for visibility
    if [ $((ELAPSED % 60)) -eq 0 ]; then
        echo "  Waiting... ${ELAPSED}s  |  server log: $(tail -1 "${SERVER_LOG}" 2>/dev/null)"
    else
        echo "  Waiting... ${ELAPSED}s"
    fi
done
echo "Server ready after ${ELAPSED}s"
echo "=== Server log tail at startup ==="
tail -10 "${SERVER_LOG}" 2>/dev/null || echo "(no server log)"

# ── Step 3: Run the PolaRiS evaluator (octi-lab-base.sif) ────────────────────
echo "=== Running PolaRiS evaluator (${POLICY_CLIENT} client) ==="
VENV=${POLARIS_DIR}/.venv
CUDA_HOME_SYNTH=${POLARIS_DIR}/.cuda_home

apptainer exec \
    --nv \
    --bind /mmfs1 \
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
            ${EVAL_ARGS}
    "

echo "=== Evaluation complete ==="
echo "Results saved to: ${RUN_FOLDER}"
echo "Date: $(date)"
