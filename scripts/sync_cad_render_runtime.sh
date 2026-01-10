#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${RUNTIME_DIR:-$HOME/Library/Application Support/dedupcad/cad-render}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

echo "Syncing CAD render runtime to: $RUNTIME_DIR"

mkdir -p "$RUNTIME_DIR"

rsync -a --delete \
  --exclude=".venv" \
  --exclude="__pycache__" \
  "$ROOT_DIR/src" \
  "$RUNTIME_DIR/"

rsync -a --delete \
  "$ROOT_DIR/scripts/cad_render_server.py" \
  "$ROOT_DIR/scripts/run_cad_render_server.sh" \
  "$ROOT_DIR/requirements.txt" \
  "$RUNTIME_DIR/"

chmod +x "$RUNTIME_DIR/run_cad_render_server.sh"

if [[ ! -x "$RUNTIME_DIR/.venv/bin/python" || "$INSTALL_DEPS" == "1" ]]; then
  echo "Bootstrapping venv in $RUNTIME_DIR/.venv"
  python3 -m venv "$RUNTIME_DIR/.venv"
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
    "$RUNTIME_DIR/.venv/bin/pip" install -r "$RUNTIME_DIR/requirements.txt"
fi

echo "Runtime sync complete."
