# DedupCAD Vision /api/compare Fallback E2E (2025-12-31)

## Scope

- Validate dedupcad-vision ML client fallback to `POST /api/compare` when `/api/v1/vectors/search` does not return the candidate hash.

## Setup

- cad-ml-platform: http://localhost:8001
- dedupcad-vision ML client: local venv, `ML_PLATFORM_URL=http://localhost:8001`

## Test Steps

1. Start cad-ml-platform on port 8001.
2. Register 100 vectors identical to the query vector, plus 1 candidate vector with low similarity.
3. Confirm `/api/v1/vectors/search` (k=100) does **not** return the candidate id.
4. Run dedupcad-vision `MLPlatformClient.compare_features` for the candidate hash.

## Commands

```bash
.venv/bin/uvicorn src.main:app --port 8001
```

```bash
.venv/bin/python - <<'PY'
import httpx

base_url = "http://localhost:8001"
headers = {"X-API-Key": "test"}

query_vec = [1.0, 0.0, 0.0, 0.0]

with httpx.Client(base_url=base_url, headers=headers, timeout=10) as client:
    for idx in range(100):
        vid = f"vec-sim-{idx:03d}"
        resp = client.post(
            "/api/v1/vectors/register",
            json={"id": vid, "vector": query_vec},
        )
        resp.raise_for_status()
    candidate_id = "a" * 64
    candidate_vec = [0.0, 1.0, 0.0, 0.0]
    resp = client.post(
        "/api/v1/vectors/register",
        json={"id": candidate_id, "vector": candidate_vec},
    )
    resp.raise_for_status()

    search = client.post(
        "/api/v1/vectors/search",
        json={"vector": query_vec, "k": 100},
    )
    search.raise_for_status()
    result_ids = [item.get("id") for item in search.json().get("results", [])]

print("candidate_id", candidate_id)
print("search_total", len(result_ids))
print("candidate_in_results", candidate_id in result_ids)
PY
```

```bash
ML_PLATFORM_URL=http://localhost:8001 /Users/huazhou/Downloads/Github/dedupcad-vision/.venv/bin/python - <<'PY'
import asyncio

from caddedup_vision.ml.client import MLPlatformClient


async def main():
    client = MLPlatformClient()
    await client.start()
    ready = await client.wait_for_ready(timeout=10)
    print("ready", ready)
    result = await client.compare_features([1.0, 0.0, 0.0, 0.0], "a" * 64)
    if result is None:
        print("result", None)
    else:
        print("similarity", getattr(result, "similarity", None))
        print("feature_distance", getattr(result, "feature_distance", None))
        print("category_match", getattr(result, "category_match", None))
        print("ocr_match", getattr(result, "ocr_match", None))
    await client.stop()


asyncio.run(main())
PY
```

## Result

- `candidate_in_results: False` (vector search misses candidate)
- `compare_features` returned similarity `0.0`, `feature_distance 1.0`
- Confirms fallback path to `/api/compare` is functional when candidate id is not in Top-K results.
