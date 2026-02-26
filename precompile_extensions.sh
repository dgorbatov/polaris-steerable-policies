#!/bin/bash
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt-all
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/mmfs1/gscratch/weirdlab/dg20/slurm_logs/precompile_%j.out
#SBATCH --error=/mmfs1/gscratch/weirdlab/dg20/slurm_logs/precompile_%j.err
#SBATCH --job-name=precompile_ext

# Pre-compiles diff-surfel-rasterization and simple-knn CUDA extensions.
# Finds nvcc from host CUDA toolkit (module system or standard paths),
# bind-mounts it into the apptainer container, and runs compilation.
# Compiled .so files are cached in ~/.cache/torch_extensions/.

set -e
echo "=== Pre-compiling CUDA extensions ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
nvidia-smi | head -5

export UV=/mmfs1/gscratch/weirdlab/dg20/UV/bin/uv
export UV_CACHE_DIR=/mmfs1/gscratch/weirdlab/dg20/UV/cache
export APPTAINER_SIF=/mmfs1/gscratch/weirdlab/dg20/Isaac/octi-lab-base.sif
export POLARIS_DIR=/mmfs1/gscratch/weirdlab/dg20/polaris
export VENV=${POLARIS_DIR}/.venv

# Step 1: Find nvcc via module system or standard paths
echo "=== Step 1: Locating nvcc ==="

# Try Lmod module system (Hyak/Klone)
for LMOD_INIT in \
    /usr/share/lmod/lmod/init/bash \
    /mmfs1/sw/lmod/lmod/init/bash \
    /etc/profile.d/lmod.sh \
    /etc/profile.d/modules.sh; do
    if [ -f "$LMOD_INIT" ]; then
        echo "Sourcing Lmod from: $LMOD_INIT"
        source "$LMOD_INIT" 2>/dev/null || true
        break
    fi
done

# Try loading CUDA module
for CUDA_MOD in cuda/13.0 cuda/13 cuda cuda/12.9 cuda/12.6 cuda/12.3; do
    if module load "$CUDA_MOD" 2>/dev/null; then
        echo "Loaded module: $CUDA_MOD"
        break
    fi
done 2>/dev/null || true

# Find nvcc via PATH or standard locations
NVCC_PATH=$(which nvcc 2>/dev/null || true)
if [ -z "$NVCC_PATH" ]; then
    for p in \
        /usr/local/cuda/bin/nvcc \
        /usr/local/cuda-13.0/bin/nvcc \
        /usr/local/cuda-12.9/bin/nvcc \
        /usr/local/cuda-12.6/bin/nvcc \
        /usr/local/cuda-12.3/bin/nvcc \
        /usr/local/cuda-11.8/bin/nvcc; do
        if [ -x "$p" ]; then
            NVCC_PATH=$p
            break
        fi
    done
fi

if [ -z "$NVCC_PATH" ]; then
    echo "ERROR: nvcc not found. Tried module system and standard paths."
    echo "Available CUDA paths:"
    ls /usr/local/ | grep cuda || echo "  none"
    exit 1
fi

HOST_CUDA_HOME=$(dirname $(dirname $NVCC_PATH))
echo "Found nvcc: $NVCC_PATH"
echo "HOST CUDA_HOME: $HOST_CUDA_HOME"
$NVCC_PATH --version

# Step 2: Build synthetic CUDA_HOME:
#   bin/nvcc   → host nvcc (CUDA 12.x, for compilation)
#   include/   → host CUDA headers (for nvcc to use during compilation)
#   lib64/     → polaris venv's CUDA 13 libs (so linker produces libcudart.so.13 DT_NEEDED)
# This ensures the compiled .so links against libcudart.so.13 which IS
# available in the container (nvidia/cu13/lib/), not libcudart.so.12.
echo "=== Step 2: Building synthetic CUDA_HOME ==="
CUDA_HOME_SYNTH=${POLARIS_DIR}/.cuda_home
CUDA13_LIB=${VENV}/lib/python3.11/site-packages/nvidia/cu13/lib

rm -rf ${CUDA_HOME_SYNTH}
mkdir -p ${CUDA_HOME_SYNTH}/bin ${CUDA_HOME_SYNTH}/lib64

# Symlink ALL host CUDA binaries into synthetic bin (nvcc, ptxas, cudafe++, fatbinary, etc.)
for f in ${HOST_CUDA_HOME}/bin/*; do
    [ -f "$f" -o -L "$f" ] && ln -sf "$f" "${CUDA_HOME_SYNTH}/bin/$(basename $f)"
done
echo "Symlinked $(ls ${CUDA_HOME_SYNTH}/bin/ | wc -l) binaries from ${HOST_CUDA_HOME}/bin/"

# In CUDA 12.x, cicc lives in nvvm/bin/ (not bin/)
if [ -d "${HOST_CUDA_HOME}/nvvm/bin" ]; then
    mkdir -p ${CUDA_HOME_SYNTH}/nvvm/bin
    for f in ${HOST_CUDA_HOME}/nvvm/bin/*; do
        [ -f "$f" -o -L "$f" ] && ln -sf "$f" "${CUDA_HOME_SYNTH}/nvvm/bin/$(basename $f)"
    done
    echo "Symlinked $(ls ${CUDA_HOME_SYNTH}/nvvm/bin/ | wc -l) binaries from ${HOST_CUDA_HOME}/nvvm/bin/"
fi

# CUDA 12 include headers (used by nvcc during compilation)
ln -sf ${HOST_CUDA_HOME}/include ${CUDA_HOME_SYNTH}/include
# CUDA 13 runtime libs (used by linker → DT_NEEDED: libcudart.so.13)
ln -sf ${CUDA13_LIB}/libcudart.so.13    ${CUDA_HOME_SYNTH}/lib64/libcudart.so.13
ln -sf ${CUDA13_LIB}/libcudart.so.13    ${CUDA_HOME_SYNTH}/lib64/libcudart.so
[ -f "${CUDA13_LIB}/libcudart_static.a" ] && ln -sf ${CUDA13_LIB}/libcudart_static.a ${CUDA_HOME_SYNTH}/lib64/libcudart_static.a

echo "Synthetic CUDA_HOME at: ${CUDA_HOME_SYNTH}"
ls -la ${CUDA_HOME_SYNTH}/bin/ ${CUDA_HOME_SYNTH}/lib64/

# Step 3: Ensure ninja build tool is in polaris venv
echo "=== Step 3: Installing ninja build tool ==="
apptainer exec \
    --nv \
    --bind /mmfs1 \
    --env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
    --env "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" \
    ${APPTAINER_SIF} \
    bash -c "
        ${UV} pip install --python ${VENV} ninja 2>&1
        echo 'ninja installed'
    "

# Step 4: Compile both extensions inside container using synthetic CUDA_HOME
# Note: bind-mount /sw so the nvcc symlink (→ /sw/cuda/...) resolves inside container
echo "=== Step 4: Compiling CUDA extensions inside container ==="
SW_BIND=""
[ -d "/sw" ] && SW_BIND="--bind /sw"
apptainer exec \
    --nv \
    --bind /mmfs1 \
    ${SW_BIND} \
    --env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
    --env "CUDA_HOME=${CUDA_HOME_SYNTH}" \
    --env "TORCH_CUDA_ARCH_LIST=8.6+PTX" \
    --env "PATH=${VENV}/bin:${CUDA_HOME_SYNTH}/bin:${CUDA_HOME_SYNTH}/nvvm/bin:${HOST_CUDA_HOME}/bin:${HOST_CUDA_HOME}/nvvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    ${APPTAINER_SIF} \
    bash -c "
        set -e
        echo 'CUDA_HOME: '${CUDA_HOME_SYNTH}
        echo 'nvcc version:'
        nvcc --version
        echo 'cicc location:' \$(which cicc 2>/dev/null || echo 'NOT FOUND')
        echo 'cudafe++ location:' \$(which cudafe++ 2>/dev/null || echo 'NOT FOUND')

        ${VENV}/bin/python - <<'PYEOF'
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Torch CUDA version: {torch.version.cuda}')

print('Compiling diff_surfel_rasterization...')
import diff_surfel_rasterization
print('diff_surfel_rasterization: OK')

print('Compiling simple_knn...')
import simple_knn
print('simple_knn: OK')

print('All extensions compiled successfully.')
PYEOF
    "

echo "=== Pre-compilation complete ==="
echo "Compiled extensions:"
ls ~/.cache/torch_extensions/ 2>/dev/null || echo "(no cache found)"
echo "Date: $(date)"
