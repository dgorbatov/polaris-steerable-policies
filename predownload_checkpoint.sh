#!/bin/bash
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt-all
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/mmfs1/gscratch/weirdlab/dg20/slurm_logs/predownload_%j.out
#SBATCH --error=/mmfs1/gscratch/weirdlab/dg20/slurm_logs/predownload_%j.err
#SBATCH --job-name=predownload_ckpt

# Downloads the pi05_droid_jointpos_polaris checkpoint from GCS into gscratch cache.
# Run this ONCE before eval.sh to avoid a slow first run.

set -e
echo "=== Pre-downloading π0.5 checkpoint ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

export UV=/mmfs1/gscratch/weirdlab/dg20/UV/bin/uv
export UV_CACHE_DIR=/mmfs1/gscratch/weirdlab/dg20/UV/cache
export APPTAINER_SIF=/mmfs1/gscratch/weirdlab/dg20/Isaac/octi-lab-base.sif
export OPENPI_DIR=/mmfs1/gscratch/weirdlab/dg20/polaris/third_party/openpi
export OPENPI_DATA_HOME=/mmfs1/gscratch/weirdlab/dg20/openpi_cache
export CHECKPOINT_URL="gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris"

mkdir -p ${OPENPI_DATA_HOME}

echo "Downloading checkpoint: ${CHECKPOINT_URL}"
echo "Destination: ${OPENPI_DATA_HOME}"

apptainer exec \
    --nv \
    --bind /mmfs1 \
    --env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
    --env "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" \
    --env "REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt" \
    --env "CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt" \
    --env "OPENPI_DATA_HOME=${OPENPI_DATA_HOME}" \
    --env "NO_GCE_CHECK=true" \
    --env "GCSFS_DEFAULT_TOKEN=anon" \
    ${APPTAINER_SIF} \
    bash -c "
        set -e
        cd ${OPENPI_DIR}
        ${UV} run python -c \"
import logging
logging.basicConfig(level=logging.INFO, force=True)
from openpi.shared.download import maybe_download
path = maybe_download('${CHECKPOINT_URL}')
print(f'Checkpoint cached at: {path}')
import os
total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
print(f'Total size: {total / 1024**3:.2f} GiB')
\"
    "

echo "=== Pre-download complete ==="
echo "Date: $(date)"
