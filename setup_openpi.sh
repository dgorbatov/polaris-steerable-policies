#!/bin/bash
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt-all
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/mmfs1/gscratch/weirdlab/dg20/slurm_logs/setup_openpi_%j.out
#SBATCH --error=/mmfs1/gscratch/weirdlab/dg20/slurm_logs/setup_openpi_%j.err
#SBATCH --job-name=setup_openpi

set -e
echo "=== Setting up openpi environment ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

export UV=/mmfs1/gscratch/weirdlab/dg20/UV/bin/uv
export UV_CACHE_DIR=/mmfs1/gscratch/weirdlab/dg20/UV/cache
export APPTAINER_SIF=/mmfs1/gscratch/weirdlab/dg20/Isaac/octi-lab-base.sif
export POLARIS_DIR=/mmfs1/gscratch/weirdlab/dg20/polaris
export OPENPI_DIR=${POLARIS_DIR}/third_party/openpi
export CONTAINER_PYTHON=/isaac-sim/kit/python/bin/python3

# Step 1: Create .venv using the container's Python (glibc 2.39)
echo "=== Step 1: Creating openpi venv with container Python ==="
apptainer exec \
    --nv \
    --bind /mmfs1 \
    --env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
    --env "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" \
    ${APPTAINER_SIF} \
    bash -c "
        cd ${OPENPI_DIR}
        UV_PYTHON=${CONTAINER_PYTHON} ${UV} venv --clear
        echo 'Venv Python platform:'
        ${OPENPI_DIR}/.venv/bin/python -c 'import platform; print(platform.libc_ver())'
    "

# Step 2: Configure git to bypass LFS (git-lfs binary is not in the container,
# and lerobot uses LFS — we configure git to use 'cat' as the filter so
# checkout just stores LFS pointers rather than failing)
echo "=== Step 2: Configuring git to bypass LFS ==="
apptainer exec \
    --nv \
    --bind /mmfs1 \
    ${APPTAINER_SIF} \
    bash -c "
        git config --global filter.lfs.smudge cat
        git config --global filter.lfs.process cat
        git config --global filter.lfs.clean cat
        git config --global filter.lfs.required false
        echo 'Git LFS bypass configured:'
        git config --global --list | grep filter.lfs
    "

# Step 3: Run uv sync inside container (installs all openpi dependencies)
echo "=== Step 3: Running uv sync for openpi ==="
apptainer exec \
    --nv \
    --bind /mmfs1 \
    --env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
    --env "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" \
    --env "REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt" \
    --env "CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt" \
    --env "GIT_LFS_SKIP_SMUDGE=1" \
    ${APPTAINER_SIF} \
    bash -c "
        set -e
        cd ${OPENPI_DIR}
        GIT_LFS_SKIP_SMUDGE=1 ${UV} sync 2>&1
        echo '=== uv sync complete ==='
    "

# Step 4: Install openpi package in editable mode
echo "=== Step 4: Installing openpi in editable mode ==="
apptainer exec \
    --nv \
    --bind /mmfs1 \
    --env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
    --env "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" \
    --env "GIT_LFS_SKIP_SMUDGE=1" \
    ${APPTAINER_SIF} \
    bash -c "
        set -e
        cd ${OPENPI_DIR}
        GIT_LFS_SKIP_SMUDGE=1 ${UV} pip install -e . 2>&1
        echo '=== editable install complete ==='
    "

# Step 5: Verify
echo "=== Step 5: Verifying openpi install ==="
apptainer exec \
    --nv \
    --bind /mmfs1 \
    ${APPTAINER_SIF} \
    bash -c "
        ${OPENPI_DIR}/.venv/bin/python -c 'import openpi; print(\"openpi: ok\")' 2>&1 || echo 'openpi import failed'
        ${OPENPI_DIR}/.venv/bin/python -c 'import jax; print(\"jax:\", jax.__version__)' 2>&1 || echo 'jax: not found'
        ${OPENPI_DIR}/.venv/bin/python -c 'import torch; print(\"torch:\", torch.__version__)' 2>&1 || echo 'torch: not found'
    "

echo "=== openpi setup complete ==="
echo "Date: $(date)"
