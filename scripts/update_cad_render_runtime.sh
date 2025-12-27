#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${CAD_RENDER_LABEL:-com.dedupcad.cad-render}"
PLIST_PATH="${CAD_RENDER_PLIST_PATH:-$HOME/Library/LaunchAgents/${LABEL}.plist}"
PORT="${CAD_RENDER_PORT:-18002}"
HEALTH_URL="${CAD_RENDER_HEALTH_URL:-http://localhost:${PORT}/health}"
RUNTIME_DIR="${RUNTIME_DIR:-$HOME/Library/Application Support/dedupcad/cad-render}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
KEEP_BACKUP="${KEEP_BACKUP:-0}"
BACKUP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cad_render_backup.XXXXXX")"

backup_runtime() {
  echo "Backing up runtime to: $BACKUP_DIR"
  if [[ -d "$RUNTIME_DIR/src" ]]; then
    mkdir -p "$BACKUP_DIR/src"
    rsync -a "$RUNTIME_DIR/src/" "$BACKUP_DIR/src/"
  fi

  for rel_path in scripts/cad_render_server.py scripts/run_cad_render_server.sh requirements.txt; do
    if [[ -f "$RUNTIME_DIR/$rel_path" ]]; then
      mkdir -p "$BACKUP_DIR/$(dirname "$rel_path")"
      cp "$RUNTIME_DIR/$rel_path" "$BACKUP_DIR/$rel_path"
    fi
  done
}

restore_runtime() {
  echo "Rolling back runtime from: $BACKUP_DIR"
  if [[ -d "$BACKUP_DIR/src" ]]; then
    mkdir -p "$RUNTIME_DIR/src"
    rsync -a "$BACKUP_DIR/src/" "$RUNTIME_DIR/src/"
  fi

  for rel_path in scripts/cad_render_server.py scripts/run_cad_render_server.sh requirements.txt; do
    if [[ -f "$BACKUP_DIR/$rel_path" ]]; then
      mkdir -p "$RUNTIME_DIR/$(dirname "$rel_path")"
      cp "$BACKUP_DIR/$rel_path" "$RUNTIME_DIR/$rel_path"
    fi
  done
}

restart_launchagent() {
  echo "Restart LaunchAgent: $LABEL"
  launchctl bootout gui/$(id -u) "$PLIST_PATH" >/dev/null 2>&1 || true
  launchctl bootstrap gui/$(id -u) "$PLIST_PATH"
  launchctl enable gui/$(id -u)/"$LABEL"
  launchctl kickstart -k gui/$(id -u)/"$LABEL"
}

wait_for_health() {
  echo "Health check: $HEALTH_URL"
  local status="000"
  for _ in $(seq 1 20); do
    status=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" || true)
    if [[ "$status" == "200" ]]; then
      echo "Health OK."
      return 0
    fi
    sleep 1
  done
  echo "Health check failed ($status): $HEALTH_URL" >&2
  return 1
}

backup_runtime

echo "Sync runtime files..."
RUNTIME_DIR="$RUNTIME_DIR" INSTALL_DEPS="$INSTALL_DEPS" \
  "$ROOT_DIR/scripts/sync_cad_render_runtime.sh"

restart_launchagent
if ! wait_for_health; then
  restore_runtime
  restart_launchagent
  wait_for_health
fi

if [[ "${RUN_ATHENA_SMOKE:-0}" == "1" ]]; then
  ATHENA_ROOT="${ATHENA_ROOT:-$HOME/Downloads/Github/Athena}"
  SMOKE_SCRIPT="$ATHENA_ROOT/scripts/smoke_test_cad_preview.sh"
  if [[ -x "$SMOKE_SCRIPT" ]]; then
    echo "Running Athena smoke test..."
    RENDER_LOG_PATH="${RENDER_LOG_PATH:-$HOME/Library/Logs/cad_render_server.log}" \
      "$SMOKE_SCRIPT"
  else
    echo "Athena smoke script not found at $SMOKE_SCRIPT, skipping."
  fi
fi

if [[ "$KEEP_BACKUP" != "1" ]]; then
  rm -rf "$BACKUP_DIR"
fi
