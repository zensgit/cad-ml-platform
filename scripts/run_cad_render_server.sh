#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
PORT="${CAD_RENDER_PORT:-18002}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at $PYTHON_BIN" >&2
  echo "Create a venv: python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export DWG_CONVERTER="${DWG_CONVERTER:-auto}"

if [[ -z "${ODA_FILE_CONVERTER_EXE:-}" ]]; then
  if [[ -x "/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter" ]]; then
    export ODA_FILE_CONVERTER_EXE="/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter"
  fi
fi

exec "$PYTHON_BIN" -m uvicorn scripts.cad_render_server:app --host 0.0.0.0 --port "$PORT"
