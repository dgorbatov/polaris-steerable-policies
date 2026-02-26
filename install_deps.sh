#!/bin/bash
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt-all
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/mmfs1/gscratch/weirdlab/dg20/slurm_logs/install_deps_%j.out
#SBATCH --error=/mmfs1/gscratch/weirdlab/dg20/slurm_logs/install_deps_%j.err
#SBATCH --job-name=polaris_install

set -e
echo "=== Starting PolaRiS dependency installation ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
nvidia-smi | head -10

export UV_CACHE_DIR=/mmfs1/gscratch/weirdlab/dg20/UV/cache
export UV=/mmfs1/gscratch/weirdlab/dg20/UV/bin/uv
export APPTAINER_SIF=/mmfs1/gscratch/weirdlab/dg20/Isaac/octi-lab-base.sif
export POLARIS_DIR=/mmfs1/gscratch/weirdlab/dg20/polaris
# Container's Python 3.11 (glibc 2.39) - required for manylinux_2_35 package compatibility
export CONTAINER_PYTHON=/isaac-sim/kit/python/bin/python3

# Step 1: Create .venv using the container's Python (glibc 2.39)
# This ensures uv sees manylinux_2_39 and can install manylinux_2_35 packages
echo "=== Creating venv with container Python (glibc 2.39) ==="
apptainer exec \
    --nv \
    --bind /mmfs1 \
    --env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
    --env "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" \
    ${APPTAINER_SIF} \
    bash -c "
        cd ${POLARIS_DIR}
        UV_PYTHON=${CONTAINER_PYTHON} ${UV} venv --clear
        echo 'Venv Python platform:'
        ${POLARIS_DIR}/.venv/bin/python -c 'import platform; print(platform.libc_ver())'
    "

# Step 2: Install setuptools<72 into the venv (needed for flatdict build)
echo "=== Installing setuptools<72 into venv ==="
apptainer exec \
    --nv \
    --bind /mmfs1 \
    --env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
    --env "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" \
    ${APPTAINER_SIF} \
    bash -c "
        cd ${POLARIS_DIR}
        ${UV} pip install --python ${POLARIS_DIR}/.venv 'setuptools<72'
    "

# Step 3: Run uv sync - installs all project dependencies into the venv
# uv will use .venv's Python (glibc 2.39) for platform detection
echo "=== Running uv sync (this may take a while - downloading many packages) ==="
apptainer exec \
    --nv \
    --bind /mmfs1 \
    --env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
    --env "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" \
    --env "REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt" \
    --env "CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt" \
    ${APPTAINER_SIF} \
    bash -c "
        set -e
        cd ${POLARIS_DIR}
        echo 'Platform check from venv Python:'
        ${POLARIS_DIR}/.venv/bin/python -c 'import platform; print(platform.libc_ver())'
        ${UV} sync 2>&1
        echo '=== uv sync complete ==='
    "

# Step 4: Verify installation
echo "=== Verifying installation ==="
apptainer exec \
    --nv \
    --bind /mmfs1 \
    ${APPTAINER_SIF} \
    bash -c "
        cd ${POLARIS_DIR}
        echo 'Installed packages (selected):'
        ${POLARIS_DIR}/.venv/bin/python -c 'import isaaclab; print(\"isaaclab:\", isaaclab.__version__)' 2>&1 || echo 'isaaclab: needs Isaac Sim context'
        ${POLARIS_DIR}/.venv/bin/python -c 'import torch; print(\"torch:\", torch.__version__)' 2>&1 || echo 'torch: not available'
        ${POLARIS_DIR}/.venv/bin/python -c 'import polaris; print(\"polaris: ok\")' 2>&1 || echo 'polaris: not found'
    "

echo "=== All done! ==="
echo "Date: $(date)"
