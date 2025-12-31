# DedupCAD Vision Compatibility & Fallback Metrics (2026-01-01)

## Scope

- Record payload format metrics for Redis job submissions.
- Track legacy base64 fallback usage in the worker.
- Ensure worker imports tolerate missing ARQ dependency.
- Add tests covering fallback and payload format instrumentation.

## Changes

- Added payload format classification + metric increment for job payloads.
- Incremented `dedup2d_legacy_b64_fallback_total` when `file_bytes_b64` is used.
- Guarded worker settings to avoid import errors when `arq` is absent.

## Tests

- `pytest tests/unit/test_dedup_2d_jobs_redis.py -v`

## Results

- OK: 23 passed, 3 skipped (metrics checks skipped when `prometheus_client` is unavailable).
