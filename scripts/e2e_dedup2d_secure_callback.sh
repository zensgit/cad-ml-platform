#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

require_bin() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required binary: $1" >&2
    exit 1
  fi
}

require_bin docker
require_bin curl
require_bin python3
require_bin cloudflared

if ! docker compose version >/dev/null 2>&1; then
  echo "Missing docker compose plugin" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d /tmp/dedup2d_secure_callback.XXXXXX)"
CALLBACK_PORT="${DEDUP2D_CALLBACK_PORT:-19080}"
CALLBACK_SECRET="${DEDUP2D_CALLBACK_HMAC_SECRET:-dedup2d-test}"
CLEANUP_TMP="${DEDUP2D_SECURE_SMOKE_CLEANUP:-0}"
MINIO_BUCKET="${DEDUP2D_S3_BUCKET:-dedup2d-uploads}"
VISION_BUCKET="${DEDUPCAD_VISION_S3_BUCKET:-dedupcad-drawings}"
VISION_START="${DEDUPCAD_VISION_START:-0}"
VISION_IMAGE="${DEDUPCAD_VISION_IMAGE:-dedupcad-vision:local}"
VISION_CONTAINER="${DEDUPCAD_VISION_CONTAINER:-dedupcad-vision-api}"
VISION_URL="${DEDUPCAD_VISION_URL:-http://dedupcad-vision-api:8000}"
CAD_ML_MINIO_PORT="${CAD_ML_MINIO_PORT:-19000}"
CAD_ML_MINIO_CONSOLE_PORT="${CAD_ML_MINIO_CONSOLE_PORT:-19001}"

CLOUDFLARED_PID=""
CALLBACK_PID=""
VISION_STARTED=0

cleanup() {
  if [ -n "${CLOUDFLARED_PID}" ] && kill -0 "${CLOUDFLARED_PID}" 2>/dev/null; then
    kill "${CLOUDFLARED_PID}" >/dev/null 2>&1 || true
  fi
  if [ -n "${CALLBACK_PID}" ] && kill -0 "${CALLBACK_PID}" 2>/dev/null; then
    kill "${CALLBACK_PID}" >/dev/null 2>&1 || true
  fi
  if [ "${VISION_STARTED}" = "1" ]; then
    docker rm -f "${VISION_CONTAINER}" >/dev/null 2>&1 || true
  fi
  if [ "${CLEANUP_TMP}" = "1" ] && [ -n "${TMP_DIR}" ]; then
    case "${TMP_DIR}" in
      /tmp/dedup2d_secure_callback.*)
        rm -rf "${TMP_DIR}" || true
        ;;
      *)
        echo "Skip cleanup for unexpected TMP_DIR: ${TMP_DIR}" >&2
        ;;
    esac
  fi
}
trap cleanup EXIT

CALLBACK_APP="${TMP_DIR}/callback_server.py"
CALLBACK_LOG="${TMP_DIR}/callback_log.jsonl"
cat > "${CALLBACK_APP}" <<'PY'
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

LOG_PATH = os.environ["CALLBACK_LOG"]
PORT = int(os.environ["CALLBACK_PORT"])


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        record = {
            "path": self.path,
            "headers": {k: v for k, v in self.headers.items()},
            "body": body.decode("utf-8", errors="replace"),
        }
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{\"ok\":true}")

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
PY

CALLBACK_LOG="${CALLBACK_LOG}" CALLBACK_PORT="${CALLBACK_PORT}" \
  python3 -u "${CALLBACK_APP}" > "${TMP_DIR}/callback_server.out" 2>&1 &
CALLBACK_PID=$!

cloudflared tunnel --url "http://127.0.0.1:${CALLBACK_PORT}" --no-autoupdate \
  > "${TMP_DIR}/cloudflared.log" 2>&1 &
CLOUDFLARED_PID=$!

callback_url=""
for i in $(seq 1 30); do
  callback_url=$(grep -Eo "https://[A-Za-z0-9.-]+\.trycloudflare\.com" "${TMP_DIR}/cloudflared.log" | tail -n 1 || true)
  if [ -n "${callback_url}" ]; then
    break
  fi
  sleep 1
  if [ "${i}" -eq 30 ]; then
    echo "Failed to obtain cloudflared URL. See ${TMP_DIR}/cloudflared.log" >&2
    exit 1
  fi
done

callback_host="${callback_url#https://}"
callback_endpoint="${callback_url}/hook"

if [ "${VISION_START}" = "1" ]; then
  if ! docker image inspect "${VISION_IMAGE}" >/dev/null 2>&1; then
    echo "Missing DedupCAD Vision image: ${VISION_IMAGE}" >&2
    exit 1
  fi
  docker rm -f "${VISION_CONTAINER}" >/dev/null 2>&1 || true
  docker run -d --name "${VISION_CONTAINER}" --network cad-ml-network \
    -e VISION_ENV=development \
    -e LOG_LEVEL=INFO \
    -e DATABASE_PATH=/app/data/drawings.db \
    -e S3_ENABLED=true \
    -e S3_ENDPOINT_URL=http://cad-ml-minio:9000 \
    -e S3_ACCESS_KEY=minioadmin \
    -e S3_SECRET_KEY=minioadmin \
    -e S3_BUCKET="${VISION_BUCKET}" \
    -e S3_REGION=us-east-1 \
    -e REDIS_URL=redis://cad-ml-redis:6379/0 \
    -e EVENT_BUS_ENABLED=false \
    "${VISION_IMAGE}" >/dev/null
  VISION_STARTED=1

  for i in $(seq 1 30); do
    if docker exec "${VISION_CONTAINER}" curl -sSf http://localhost:8000/health >/dev/null 2>&1; then
      break
    fi
    sleep 1
    if [ "${i}" -eq 30 ]; then
      echo "DedupCAD Vision health check timed out" >&2
      exit 1
    fi
  done

  if [ -z "${DEDUPCAD_VISION_URL:-}" ]; then
    VISION_URL="http://${VISION_CONTAINER}:8000"
  fi
fi

# Ensure buckets exist

docker run --rm --network cad-ml-network --entrypoint /bin/sh minio/mc:latest -c \
  "mc alias set myminio http://cad-ml-minio:9000 minioadmin minioadmin >/dev/null; \
   mc mb --ignore-existing myminio/${MINIO_BUCKET}; \
   mc mb --ignore-existing myminio/${VISION_BUCKET}; \
   mc anonymous set download myminio/${MINIO_BUCKET}; \
   mc anonymous set download myminio/${VISION_BUCKET}"

CAD_ML_MINIO_PORT="${CAD_ML_MINIO_PORT}" \
CAD_ML_MINIO_CONSOLE_PORT="${CAD_ML_MINIO_CONSOLE_PORT}" \
DEDUPCAD_VISION_URL="${VISION_URL}" \
DEDUP2D_CALLBACK_ALLOW_HTTP=0 \
DEDUP2D_CALLBACK_BLOCK_PRIVATE_NETWORKS=1 \
DEDUP2D_CALLBACK_RESOLVE_DNS=1 \
DEDUP2D_CALLBACK_ALLOWLIST="${callback_host}" \
DEDUP2D_CALLBACK_HMAC_SECRET="${CALLBACK_SECRET}" \
  docker compose -f deployments/docker/docker-compose.yml \
  -f deployments/docker/docker-compose.minio.yml \
  -f deployments/docker/docker-compose.dedup2d-staging.yml \
  up -d --no-deps --force-recreate cad-ml-api dedup2d-worker

for i in $(seq 1 30); do
  if curl -sSf http://localhost:8000/health > "${TMP_DIR}/health.json"; then
    break
  fi
  sleep 1
  if [ "${i}" -eq 30 ]; then
    echo "cad-ml-api health check timed out" >&2
    exit 1
  fi
done

curl -sSf -H "X-API-Key: test" http://localhost:8000/api/v1/dedup/2d/health > "${TMP_DIR}/dedup2d_health.json"

curl -sSf http://localhost:8000/metrics/ > "${TMP_DIR}/metrics.txt"
if ! grep -q "dedup2d_jobs_total" "${TMP_DIR}/metrics.txt"; then
  echo "dedup2d metrics missing from /metrics/" >&2
  exit 1
fi

http_code=$(curl -s -o "${TMP_DIR}/reject.json" -w "%{http_code}" -X POST \
  "http://localhost:8000/api/v1/dedup/2d/search?mode=balanced&max_results=1&async=true&callback_url=https://example.com/hook" \
  -H "X-API-Key: test" \
  -F "file=@data/dedupcad_batch_demo/test_left.png;type=image/png")

if [ "${http_code}" != "400" ]; then
  echo "Expected 400 for unallowlisted callback, got ${http_code}" >&2
  exit 1
fi

submit_resp=$(curl -sSf -X POST \
  "http://localhost:8000/api/v1/dedup/2d/search?mode=balanced&max_results=5&async=true&callback_url=${callback_endpoint}" \
  -H "X-API-Key: test" \
  -F "file=@data/dedupcad_batch_demo/test_left.png;type=image/png")

echo "${submit_resp}" > "${TMP_DIR}/submit.json"
job_id=$(TMP_DIR="${TMP_DIR}" python3 - <<'PY'
import json
import os

with open(os.path.join(os.environ["TMP_DIR"], "submit.json"), "r", encoding="utf-8") as fh:
    data = json.load(fh)
print(data.get("job_id", ""))
PY
)

if [ -z "${job_id}" ]; then
  echo "Missing job_id in response" >&2
  exit 1
fi

job_status=""
for i in $(seq 1 30); do
  job_resp=$(curl -sSf -H "X-API-Key: test" "http://localhost:8000/api/v1/dedup/2d/jobs/${job_id}")
  echo "${job_resp}" > "${TMP_DIR}/job.json"
  job_status=$(TMP_DIR="${TMP_DIR}" python3 - <<'PY'
import json
import os

with open(os.path.join(os.environ["TMP_DIR"], "job.json"), "r", encoding="utf-8") as fh:
    data = json.load(fh)
print(data.get("status", ""))
PY
)
  if [ "${job_status}" = "completed" ] || [ "${job_status}" = "failed" ]; then
    break
  fi
  sleep 2
  if [ "${i}" -eq 30 ]; then
    echo "Job polling timed out" >&2
    exit 1
  fi
done

if [ "${job_status}" != "completed" ]; then
  echo "Job did not complete successfully (status=${job_status})" >&2
  exit 1
fi

tenant_id=$(TMP_DIR="${TMP_DIR}" python3 - <<'PY'
import json
import os

with open(os.path.join(os.environ["TMP_DIR"], "job.json"), "r", encoding="utf-8") as fh:
    data = json.load(fh)
print(data.get("tenant_id", ""))
PY
)

curl -sSf -H "X-API-Key: test" "http://localhost:8000/api/v1/dedup/2d/jobs?limit=5" > "${TMP_DIR}/jobs_list.json"
TMP_DIR="${TMP_DIR}" JOB_ID="${job_id}" python3 - <<'PY'
import json
import os
import sys

with open(os.path.join(os.environ["TMP_DIR"], "jobs_list.json"), "r", encoding="utf-8") as fh:
    data = json.load(fh)
job_id = os.environ["JOB_ID"]
items = data.get("jobs", [])
if job_id not in {item.get("job_id") for item in items}:
    sys.exit(1)
PY

job_type=$(docker exec cad-ml-redis redis-cli type "dedup2d:job:${job_id}" | tr -d '\r')
payload_type=$(docker exec cad-ml-redis redis-cli type "dedup2d:payload:${job_id}" | tr -d '\r')
result_type=$(docker exec cad-ml-redis redis-cli type "dedup2d:result:${job_id}" | tr -d '\r')
tenant_type=$(docker exec cad-ml-redis redis-cli type "dedup2d:tenant:${tenant_id}:jobs" | tr -d '\r')

if [ "${job_type}" != "hash" ] || [ "${payload_type}" != "string" ] || [ "${result_type}" != "string" ] || [ "${tenant_type}" != "zset" ]; then
  echo "Unexpected Redis key types: job=${job_type} payload=${payload_type} result=${result_type} tenant=${tenant_type}" >&2
  exit 1
fi

docker exec cad-ml-redis redis-cli get "dedup2d:payload:${job_id}" > "${TMP_DIR}/payload.json"

TMP_DIR="${TMP_DIR}" python3 - <<'PY' > "${TMP_DIR}/s3_key.txt"
import json
import os
import sys

with open(os.path.join(os.environ["TMP_DIR"], "payload.json"), "r", encoding="utf-8") as fh:
    data = json.load(fh)
file_ref = data.get("file_ref")
if not isinstance(file_ref, dict):
    sys.exit(1)
key = file_ref.get("key")
if not key:
    sys.exit(1)
print(key)
PY

s3_key="$(cat "${TMP_DIR}/s3_key.txt")"
if [ -z "${s3_key}" ]; then
  echo "Missing S3 key in payload" >&2
  exit 1
fi

if docker run --rm --network cad-ml-network --entrypoint /bin/sh minio/mc:latest -c \
  "mc alias set myminio http://cad-ml-minio:9000 minioadmin minioadmin >/dev/null; \
   mc stat myminio/${MINIO_BUCKET}/${s3_key}" > "${TMP_DIR}/mc_stat.txt" 2>&1; then
  echo "S3 object still exists: ${s3_key}" >&2
  exit 1
fi

if ! grep -qi "Object does not exist" "${TMP_DIR}/mc_stat.txt"; then
  echo "Unexpected mc stat error" >&2
  cat "${TMP_DIR}/mc_stat.txt" >&2
  exit 1
fi

CALLBACK_SECRET="${CALLBACK_SECRET}" CALLBACK_LOG="${CALLBACK_LOG}" python3 - <<'PY'
import json
import hmac
import hashlib
import os
import sys

secret = os.environ["CALLBACK_SECRET"].encode("utf-8")
log_path = os.environ["CALLBACK_LOG"]

with open(log_path, "r", encoding="utf-8") as fh:
    lines = fh.readlines()
if not lines:
    sys.exit(1)
record = json.loads(lines[-1])
headers = record.get("headers", {})
body = record.get("body", "").encode("utf-8")
job_id = headers.get("X-Dedup-Job-Id", "")
sig_header = headers.get("X-Dedup-Signature", "")
if not job_id or not sig_header:
    sys.exit(1)
parts = dict(item.split("=", 1) for item in sig_header.split(","))
ts = int(parts["t"])
msg = f"{ts}.{job_id}.".encode("utf-8") + body
sig = hmac.new(secret, msg, hashlib.sha256).hexdigest()
if sig != parts.get("v1"):
    sys.exit(1)
PY

cat > "${TMP_DIR}/summary.txt" <<EOF
callback_url=${callback_endpoint}
callback_host=${callback_host}
job_id=${job_id}
tenant_id=${tenant_id}
job_status=${job_status}
artifacts=${TMP_DIR}
EOF

echo "Secure callback smoke test succeeded. Artifacts: ${TMP_DIR}"
