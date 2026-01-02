# DEV_DEDUP2D_S3_LIFECYCLE_20260101

## Scope
- Validate S3/MinIO upload + cleanup lifecycle for dedup2d async jobs.

## Method
Executed inside `cad-ml-api` container to reuse configured MinIO credentials:
```bash
docker exec -i cad-ml-api python - <<'PY'
import base64
import os
import time
import requests
import boto3

png_b64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)
img_path = "/tmp/dedup2d_smoke.png"
with open(img_path, "wb") as f:
    f.write(base64.b64decode(png_b64))

bucket = os.getenv("DEDUP2D_S3_BUCKET")
prefix = (os.getenv("DEDUP2D_S3_PREFIX") or "").strip("/")
endpoint = os.getenv("DEDUP2D_S3_ENDPOINT")
region = os.getenv("DEDUP2D_S3_REGION")
access = os.getenv("AWS_ACCESS_KEY_ID")
secret = os.getenv("AWS_SECRET_ACCESS_KEY")

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    region_name=region,
    aws_access_key_id=access,
    aws_secret_access_key=secret,
)

def list_keys():
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [obj["Key"] for obj in resp.get("Contents", [])]

before_keys = list_keys()
print(f"before_objects={len(before_keys)}")

with open(img_path, "rb") as f:
    resp = requests.post(
        "http://localhost:8000/api/v1/dedup/2d/search?async=true",
        headers={"X-API-Key": "test"},
        files={"file": ("smoke.png", f, "image/png")},
        timeout=10,
    )
resp.raise_for_status()
job_id = resp.json()["job_id"]
print(f"job_id={job_id}")

observed_keys = []
for i in range(60):
    keys = list_keys()
    if keys:
        observed_keys = keys
        print(f"observed_objects={len(keys)}")
        break
    time.sleep(1)

status = "unknown"
for i in range(90):
    job_resp = requests.get(
        f"http://localhost:8000/api/v1/dedup/2d/jobs/{job_id}",
        headers={"X-API-Key": "test"},
        timeout=10,
    )
    job_resp.raise_for_status()
    status = job_resp.json().get("status")
    if status in {"completed", "failed", "canceled"}:
        print(f"final_status={status}")
        break
    time.sleep(1)

for _ in range(3):
    time.sleep(1)

after_keys = list_keys()
print(f"after_objects={len(after_keys)}")
if observed_keys:
    print("observed_keys_sample=", observed_keys[:3])
PY
```

## Observed Output (summary)
- `before_objects=0`
- `observed_objects=1` (key under `uploads/<job_id>/..._smoke.png`)
- `final_status=completed`
- `after_objects=0`

## Notes
- `boto3` emitted a Python 3.9 deprecation warning (non-blocking).

## Result
- S3/MinIO upload observed during job execution.
- Cleanup confirmed via zero objects after completion.
