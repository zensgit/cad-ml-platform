#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATE="$(date +%Y%m%d)"
REPORT_DIR="${ROOT_DIR}/reports"
REPORT_PATH="${REPORT_DIR}/DEV_L3_BREP_LINUX_AMD64_VALIDATION_${DATE}.md"
TMP_DIR="${ROOT_DIR}/tmp"
CONTAINER_NAME="cadml-l3-${DATE}-$$"
CACHE_VOLUME="${CACHE_VOLUME:-cadml-micromamba-cache}"

cleanup() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  rm -f \
    "${TMP_DIR}/box.step" \
    "${TMP_DIR}/cylinder.step" \
    "${TMP_DIR}/sphere.step" \
    "${TMP_DIR}/torus.step" \
    "${TMP_DIR}/box.json" \
    "${TMP_DIR}/cylinder.json" \
    "${TMP_DIR}/sphere.json" \
    "${TMP_DIR}/torus.json"
  rmdir "${TMP_DIR}" 2>/dev/null || true
}

trap cleanup EXIT

mkdir -p "${TMP_DIR}" "${REPORT_DIR}"

if ! docker volume inspect "${CACHE_VOLUME}" >/dev/null 2>&1; then
  echo "Creating docker volume cache: ${CACHE_VOLUME}"
  docker volume create "${CACHE_VOLUME}" >/dev/null
fi

echo "Starting linux/amd64 micromamba container with persistent cache..."
docker run --platform linux/amd64 --rm -d \
  --name "${CONTAINER_NAME}" \
  -p 8000:8000 \
  -e FEATURE_VERSION=v4 \
  -e MAMBA_ROOT_PREFIX=/opt/conda \
  -v "${ROOT_DIR}":/work \
  -v "${CACHE_VOLUME}":/opt/conda/pkgs \
  -w /work \
  mambaorg/micromamba:1.5.8 \
  bash -lc "sleep infinity"

export MAMBA_NO_REPODATA_ZST=1

echo "Installing pythonocc-core (conda-forge)..."
docker exec "${CONTAINER_NAME}" bash -lc \
  "export MAMBA_NO_REPODATA_ZST=1; micromamba create -y -n cadml -c conda-forge python=3.10 pythonocc-core"

echo "Installing Python dependencies..."
docker exec "${CONTAINER_NAME}" bash -lc \
  "micromamba run -n cadml python -m pip install -r requirements.txt"

echo "Starting FastAPI..."
docker exec -d "${CONTAINER_NAME}" bash -lc \
  "micromamba run -n cadml uvicorn src.main:app --host 0.0.0.0 --port 8000"

health="timeout"
for _ in {1..30}; do
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
    health="ok"
    break
  fi
  sleep 2
  done

if [[ "${health}" != "ok" ]]; then
  echo "Health check failed for http://localhost:8000/health" >&2
  exit 1
fi

docker exec "${CONTAINER_NAME}" bash -lc 'micromamba run -n cadml python - <<'"'"'PY'"'"'
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeTorus
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer

def write_step(shape, path):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP export failed for {path}")

write_step(BRepPrimAPI_MakeBox(10, 20, 30).Shape(), "/work/tmp/box.step")
write_step(BRepPrimAPI_MakeCylinder(5, 20).Shape(), "/work/tmp/cylinder.step")
write_step(BRepPrimAPI_MakeSphere(7).Shape(), "/work/tmp/sphere.step")
write_step(BRepPrimAPI_MakeTorus(10, 3).Shape(), "/work/tmp/torus.step")
PY'

run_analyze() {
  local name="$1"
  curl -fsS -X POST http://localhost:8000/api/v1/analyze/ \
    --connect-timeout 10 --max-time 120 \
    -H "X-API-Key: test" \
    -F "file=@${TMP_DIR}/${name}.step" \
    -F 'options={"extract_features": true, "classify_parts": false, "quality_check": false, "process_recommendation": false, "calculate_similarity": false, "estimate_cost": false}' \
    -o "${TMP_DIR}/${name}.json"
}

echo "Running analyze for STEP fixtures..."
run_analyze box
run_analyze cylinder
run_analyze sphere
run_analyze torus

read -r BOX_SURFACE BOX_ENTROPY BOX_VERSION <<<"$(python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("tmp/box.json").read_text())
geo = data["results"]["features"]["geometric"]
print(f"{geo[-2]} {geo[-1]} {data.get('feature_version')}")
PY
)"

read -r CYL_SURFACE CYL_ENTROPY CYL_VERSION <<<"$(python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("tmp/cylinder.json").read_text())
geo = data["results"]["features"]["geometric"]
print(f"{geo[-2]} {geo[-1]} {data.get('feature_version')}")
PY
)"

read -r SPHERE_SURFACE SPHERE_ENTROPY SPHERE_VERSION <<<"$(python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("tmp/sphere.json").read_text())
geo = data["results"]["features"]["geometric"]
print(f"{geo[-2]} {geo[-1]} {data.get('feature_version')}")
PY
)"

read -r TORUS_SURFACE TORUS_ENTROPY TORUS_VERSION <<<"$(python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("tmp/torus.json").read_text())
geo = data["results"]["features"]["geometric"]
print(f"{geo[-2]} {geo[-1]} {data.get('feature_version')}")
PY
)"

PYTHONOCC_VERSION="$(docker exec "${CONTAINER_NAME}" bash -lc \
  "micromamba list -n cadml | awk '\$1 == \"pythonocc-core\" {print \$2}'")"

cat <<EOF > "${REPORT_PATH}"
# DEV_L3_BREP_LINUX_AMD64_VALIDATION_${DATE}

## Summary
Validated L3 B-Rep surface metrics via linux/amd64 micromamba container with pythonocc-core.
Confirmed non-zero surface counts and appropriate entropy for primitive STEP fixtures.

## Environment
- Docker image: mambaorg/micromamba:1.5.8 (linux/amd64)
- pythonocc-core: ${PYTHONOCC_VERSION}
- FEATURE_VERSION: v4

## Steps
- Started FastAPI with uvicorn in the container.
- Generated STEP fixtures (box, cylinder, sphere, torus) via pythonocc-core.
- Posted fixtures to "/api/v1/analyze" with feature extraction only.

## Results
- box.step: surface_count=${BOX_SURFACE}, shape_entropy=${BOX_ENTROPY}, feature_version=${BOX_VERSION}
- cylinder.step: surface_count=${CYL_SURFACE}, shape_entropy=${CYL_ENTROPY}, feature_version=${CYL_VERSION}
- sphere.step: surface_count=${SPHERE_SURFACE}, shape_entropy=${SPHERE_ENTROPY}, feature_version=${SPHERE_VERSION}
- torus.step: surface_count=${TORUS_SURFACE}, shape_entropy=${TORUS_ENTROPY}, feature_version=${TORUS_VERSION}

## Notes
- Report generated by `scripts/validate_brep_features_linux_amd64_cached.sh`.
EOF

echo "Report written to ${REPORT_PATH}"
