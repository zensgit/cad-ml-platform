#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/validate_uvnet_step_docker.sh <data_dir> [batch_size] [limit]

Runs UV-Net graph STEP dry-run inside a linux/amd64 micromamba container.

Args:
  data_dir    Directory containing STEP files
  batch_size  Optional batch size (default: 2)
  limit       Optional max samples (default: 2)

Env overrides:
  UVNET_DOCKER_IMAGE      Docker image (default: mambaorg/micromamba:1.5.8)
  UVNET_CONDA_ENV         Conda env name (default: cadml)
  UVNET_PYTHON_VERSION    Python version (default: 3.10)
  UVNET_TORCH_INDEX_URL   Torch wheel index (default: https://download.pytorch.org/whl/cpu)
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

DATA_DIR="$1"
BATCH_SIZE="${2:-2}"
LIMIT="${3:-2}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not found in PATH." >&2
  exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Data dir not found: $DATA_DIR" >&2
  exit 1
fi

IMAGE="${UVNET_DOCKER_IMAGE:-mambaorg/micromamba:1.5.8}"
ENV_NAME="${UVNET_CONDA_ENV:-cadml}"
PYTHON_VERSION="${UVNET_PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${UVNET_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

WORKDIR="$(pwd)"

set -x

docker run --rm -t --platform linux/amd64 \
  -v "${WORKDIR}:/workspace" \
  -w /workspace \
  "${IMAGE}" \
  bash -lc "set -euo pipefail
micromamba create -y -n ${ENV_NAME} -c conda-forge python=${PYTHON_VERSION} pythonocc-core
micromamba run -n ${ENV_NAME} python -m pip install --index-url ${TORCH_INDEX_URL} torch
micromamba run -n ${ENV_NAME} env LD_LIBRARY_PATH=/opt/conda/envs/${ENV_NAME}/lib \
  python scripts/train_uvnet_graph_dryrun.py --data-dir '${DATA_DIR}' --batch-size ${BATCH_SIZE} --limit ${LIMIT}
"
