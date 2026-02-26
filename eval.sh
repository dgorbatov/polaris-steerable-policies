#!/bin/bash
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt-all
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/mmfs1/gscratch/weirdlab/dg20/slurm_logs/eval_%j.out
#SBATCH --error=/mmfs1/gscratch/weirdlab/dg20/slurm_logs/eval_%j.err
#SBATCH --job-name=polaris_eval

# Usage: sbatch eval.sh [--environment ENV] [--run-folder FOLDER] [--rollouts N]
# Example: sbatch eval.sh --environment DROID-FoodBussing --run-folder runs/test

set -e
echo "=== PolaRiS π0.5 Evaluation ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
nvidia-smi | head -10

export UV=/mmfs1/gscratch/weirdlab/dg20/UV/bin/uv
export UV_CACHE_DIR=/mmfs1/gscratch/weirdlab/dg20/UV/cache
export APPTAINER_SIF=/mmfs1/gscratch/weirdlab/dg20/Isaac/octi-lab-base.sif
export POLARIS_DIR=/mmfs1/gscratch/weirdlab/dg20/polaris
export OPENPI_DIR=${POLARIS_DIR}/third_party/openpi

# Checkpoint cache in gscratch (avoids filling 10GB home dir)
export OPENPI_DATA_HOME=/mmfs1/gscratch/weirdlab/dg20/openpi_cache
mkdir -p ${OPENPI_DATA_HOME}

# Policy server port
export POLICY_PORT=8000

# Eval arguments (can be passed via SLURM --export or hardcoded here)
EVAL_ENV="${EVAL_ENV:-DROID-FoodBussing}"
RUN_FOLDER="${RUN_FOLDER:-${POLARIS_DIR}/runs/eval_$(date +%Y%m%d_%H%M%S)}"
EVAL_ARGS="${EVAL_ARGS:-}"

echo "Environment: ${EVAL_ENV}"
echo "Run folder: ${RUN_FOLDER}"
mkdir -p "${RUN_FOLDER}"

# Cleanup handler: kill background processes on exit
cleanup() {
    echo "=== Cleaning up ==="
    if [ -n "${SERVER_PID}" ]; then
        echo "Killing policy server (PID ${SERVER_PID})"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    echo "Done."
}
trap cleanup EXIT

# ====================================================================
# Step 1: Start the π0.5 policy server in the background
# ====================================================================
echo "=== Starting π0.5 policy server on port ${POLICY_PORT} ==="
apptainer exec \
    --nv \
    --bind /mmfs1 \
    --env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
    --env "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" \
    --env "REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt" \
    --env "CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt" \
    --env "OPENPI_DATA_HOME=${OPENPI_DATA_HOME}" \
    --env "XLA_PYTHON_CLIENT_MEM_FRACTION=0.35" \
    --env "NO_GCE_CHECK=true" \
    --env "GCSFS_DEFAULT_TOKEN=anon" \
    ${APPTAINER_SIF} \
    bash -c "
        cd ${OPENPI_DIR}
        ${UV} run scripts/serve_policy.py \
            --port ${POLICY_PORT} \
            policy:checkpoint \
            --policy.config pi05_droid_jointpos_polaris \
            --policy.dir gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris
    " &
SERVER_PID=$!
echo "Policy server started with PID ${SERVER_PID}"

# ====================================================================
# Step 2: Wait for the policy server to become ready
# ====================================================================
echo "=== Waiting for policy server on port ${POLICY_PORT} ==="
MAX_WAIT=900  # 15 minutes (includes checkpoint download from GCS on first run)
ELAPSED=0
while ! nc -z localhost ${POLICY_PORT} 2>/dev/null; do
    if [ ${ELAPSED} -ge ${MAX_WAIT} ]; then
        echo "ERROR: Policy server did not start within ${MAX_WAIT}s"
        exit 1
    fi
    # Check if server process is still alive
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "ERROR: Policy server process died (PID ${SERVER_PID})"
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  Waiting... ${ELAPSED}s elapsed"
done
echo "Policy server is ready after ${ELAPSED}s"

# ====================================================================
# Step 3: Run the PolaRiS evaluator
# ====================================================================
echo "=== Running PolaRiS evaluator ==="
# CUDA_HOME needed for diff-surfel-rasterization JIT compilation (first run)
# or for loading pre-compiled cache (subsequent runs)
CUDA_HOME_SYNTH=${POLARIS_DIR}/.cuda_home
VENV=${POLARIS_DIR}/.venv
apptainer exec \
    --nv \
    --bind /mmfs1 \
    --env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
    --env "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" \
    --env "OMNI_KIT_ACCEPT_EULA=Y" \
    --env "CUDA_HOME=${CUDA_HOME_SYNTH}" \
    --env "TORCH_CUDA_ARCH_LIST=8.6+PTX" \
    --env "LD_LIBRARY_PATH=${VENV}/lib/python3.11/site-packages/nvidia/cu13/lib" \
    ${APPTAINER_SIF} \
    bash -c "
        cd ${POLARIS_DIR}
        ${UV} run scripts/eval.py \
            --environment ${EVAL_ENV} \
            --policy.port ${POLICY_PORT} \
            --run-folder ${RUN_FOLDER} \
            ${EVAL_ARGS}
    "

echo "=== Evaluation complete ==="
echo "Results saved to: ${RUN_FOLDER}"
echo "Date: $(date)"
