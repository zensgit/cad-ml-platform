#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATE="$(date +%Y%m%d)"
ENV_NAME="${ENV_NAME:-cad-ml-brep-m4}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.micromamba}"
MAMBA_BIN_DIR="${MAMBA_BIN_DIR:-${HOME}/.local/bin}"
MAMBA_BIN="${MAMBA_BIN_DIR}/micromamba"
MICROMAMBA_PLATFORM="${MICROMAMBA_PLATFORM:-osx-arm64}"
INSTALL_PROJECT_REQUIREMENTS="${INSTALL_PROJECT_REQUIREMENTS:-1}"
INSTALL_PYTORCH="${INSTALL_PYTORCH:-0}"
INSTALL_PYG="${INSTALL_PYG:-0}"
RUN_ONLINE_SMOKE="${RUN_ONLINE_SMOKE:-0}"
RUN_STEP_DIR_EVAL="${RUN_STEP_DIR_EVAL:-0}"
H5_FILE="${H5_FILE:-/private/tmp/cad-ai-example-data-20260307/HPSketch/data/0000/00000007_1.h5}"
STEP_FILE="${STEP_FILE:-/private/tmp/cad-ai-example-data-20260307/foxtrot/examples/cube_hole.step}"
SMOKE_OUTPUT="${SMOKE_OUTPUT:-${ROOT_DIR}/reports/experiments/${DATE}/online_example_ai_inputs_validation_micromamba.json}"
STEP_DIR="${STEP_DIR:-/private/tmp/cad-ai-example-data-20260307/foxtrot/examples}"
STEP_DIR_OUTPUT="${STEP_DIR_OUTPUT:-${ROOT_DIR}/reports/experiments/${DATE}/brep_step_dir_eval_foxtrot}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--run-smoke] [--run-step-dir-eval] [--with-pytorch] [--with-pyg]

Bootstrap a macOS ARM64 micromamba environment for pythonocc-based STEP/B-Rep
validation without requiring a preinstalled Conda distribution.

Environment variables:
  ENV_NAME                     Target environment name (default: ${ENV_NAME})
  PYTHON_VERSION               Python version (default: ${PYTHON_VERSION})
  MAMBA_ROOT_PREFIX            Micromamba root prefix (default: ${MAMBA_ROOT_PREFIX})
  MAMBA_BIN_DIR                Directory to install micromamba (default: ${MAMBA_BIN_DIR})
  INSTALL_PROJECT_REQUIREMENTS Install requirements.txt via pip (default: ${INSTALL_PROJECT_REQUIREMENTS})
  INSTALL_PYTORCH              Install pytorch+torchvision from pytorch channel (default: ${INSTALL_PYTORCH})
  INSTALL_PYG                  Install torch-geometric extras via pip (default: ${INSTALL_PYG})
  RUN_ONLINE_SMOKE             Run validate_online_example_ai_inputs.py after setup (default: ${RUN_ONLINE_SMOKE})
  RUN_STEP_DIR_EVAL            Run eval_brep_step_dir.py after setup (default: ${RUN_STEP_DIR_EVAL})
  H5_FILE                      HPSketch sample path for smoke validation
  STEP_FILE                    STEP sample path for smoke validation
  SMOKE_OUTPUT                 JSON output path for smoke validation
  STEP_DIR                     STEP directory for batch evaluation
  STEP_DIR_OUTPUT              Output directory for batch STEP evaluation

Flags:
  --run-smoke                  Set RUN_ONLINE_SMOKE=1
  --run-step-dir-eval          Set RUN_STEP_DIR_EVAL=1
  --with-pytorch               Set INSTALL_PYTORCH=1
  --with-pyg                   Set INSTALL_PYG=1 (implies --with-pytorch)
  --skip-project-requirements  Set INSTALL_PROJECT_REQUIREMENTS=0
  --help                       Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-smoke)
      RUN_ONLINE_SMOKE=1
      shift
      ;;
    --with-pytorch)
      INSTALL_PYTORCH=1
      shift
      ;;
    --run-step-dir-eval)
      RUN_STEP_DIR_EVAL=1
      shift
      ;;
    --with-pyg)
      INSTALL_PYTORCH=1
      INSTALL_PYG=1
      shift
      ;;
    --skip-project-requirements)
      INSTALL_PROJECT_REQUIREMENTS=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_command curl
require_command tar
require_command install

mkdir -p "${MAMBA_BIN_DIR}" "${MAMBA_ROOT_PREFIX}"

download_micromamba() {
  if [[ -x "${MAMBA_BIN}" ]]; then
    echo "Using existing micromamba: ${MAMBA_BIN}"
    return
  fi

  local tmpdir
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' RETURN

  echo "Downloading micromamba for ${MICROMAMBA_PLATFORM}..."
  curl -Ls "https://micro.mamba.pm/api/micromamba/${MICROMAMBA_PLATFORM}/latest" \
    -o "${tmpdir}/micromamba.tar.bz2"
  tar -xjf "${tmpdir}/micromamba.tar.bz2" -C "${tmpdir}" bin/micromamba
  install -m 0755 "${tmpdir}/bin/micromamba" "${MAMBA_BIN}"
  echo "Installed micromamba to ${MAMBA_BIN}"
}

micromamba_run() {
  "${MAMBA_BIN}" run -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" "$@"
}

create_or_update_env() {
  local packages=(
    "-c" "conda-forge"
    "python=${PYTHON_VERSION}"
    "pythonocc-core"
    "ezdxf"
    "trimesh"
    "h5py"
    "pip"
  )

  if [[ -d "${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}" ]]; then
    echo "Updating existing environment: ${ENV_NAME}"
    "${MAMBA_BIN}" install -y -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" "${packages[@]}"
  else
    echo "Creating environment: ${ENV_NAME}"
    "${MAMBA_BIN}" create -y -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" "${packages[@]}"
  fi
}

install_project_requirements() {
  if [[ "${INSTALL_PROJECT_REQUIREMENTS}" != "1" ]]; then
    echo "Skipping requirements.txt installation."
    return
  fi

  echo "Installing project requirements.txt..."
  micromamba_run python -m pip install -r "${ROOT_DIR}/requirements.txt"
}

install_optional_ml_stack() {
  if [[ "${INSTALL_PYTORCH}" == "1" ]]; then
    echo "Installing pytorch + torchvision..."
    "${MAMBA_BIN}" install -y -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" -c pytorch pytorch torchvision
  fi

  if [[ "${INSTALL_PYG}" == "1" ]]; then
    echo "Installing PyTorch Geometric extras..."
    micromamba_run python -m pip install \
      torch-geometric \
      torch-scatter \
      torch-sparse \
      torch-cluster \
      torch-spline-conv
  fi
}

run_smoke_validation() {
  if [[ "${RUN_ONLINE_SMOKE}" != "1" ]]; then
    return
  fi

  echo "Running online example smoke validation..."
  micromamba_run python "${ROOT_DIR}/scripts/validate_online_example_ai_inputs.py" \
    --h5-file "${H5_FILE}" \
    --step-file "${STEP_FILE}" \
    --output "${SMOKE_OUTPUT}"
}

run_step_dir_eval() {
  if [[ "${RUN_STEP_DIR_EVAL}" != "1" ]]; then
    return
  fi

  echo "Running STEP directory evaluation..."
  micromamba_run python "${ROOT_DIR}/scripts/eval_brep_step_dir.py" \
    --step-dir "${STEP_DIR}" \
    --output-dir "${STEP_DIR_OUTPUT}"
}

download_micromamba
create_or_update_env
install_project_requirements
install_optional_ml_stack
run_smoke_validation
run_step_dir_eval

cat <<EOF

Setup complete.

Environment:
  name: ${ENV_NAME}
  root prefix: ${MAMBA_ROOT_PREFIX}
  micromamba: ${MAMBA_BIN}

Run STEP/B-Rep smoke manually:
  ${MAMBA_BIN} run -r ${MAMBA_ROOT_PREFIX} -n ${ENV_NAME} \\
    python ${ROOT_DIR}/scripts/validate_online_example_ai_inputs.py \\
    --step-file ${STEP_FILE} \\
    --output ${SMOKE_OUTPUT}

Run STEP directory evaluation manually:
  ${MAMBA_BIN} run -r ${MAMBA_ROOT_PREFIX} -n ${ENV_NAME} \\
    python ${ROOT_DIR}/scripts/eval_brep_step_dir.py \\
    --step-dir ${STEP_DIR} \\
    --output-dir ${STEP_DIR_OUTPUT}

Check pythonocc availability:
  ${MAMBA_BIN} run -r ${MAMBA_ROOT_PREFIX} -n ${ENV_NAME} \\
    python -c 'from src.core.geometry.engine import HAS_OCC; print(HAS_OCC)'
EOF
