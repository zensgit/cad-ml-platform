# DEV_ANALYZE_BATCH_CLASSIFY_AND_V16_HEALTH_20260207

## Goal

1. Add an API endpoint to **batch classify** uploaded CAD files (DXF/DWG) in one request.
2. Expose **V16 classifier runtime health + cache controls** under `/api/v1/health/*`.

## Changes

### 1) Batch classify endpoint

- `src/api/v1/analyze.py`
  - Added `POST /api/v1/analyze/batch-classify`
    - Accepts multiple uploaded files via `files=[...]`
    - Supports: `.dxf`, `.dwg`
    - Uses V16 classifier batch path when available:
      - `PartClassifierV16.predict_batch(paths, max_workers=...)`
    - Falls back to sequential V6 when V16 is unavailable.
  - Fixed result alignment bug when mixed supported/unsupported formats are uploaded:
    - Maintain `work_items[(result_index, temp_path)]` mapping so the response order always matches the input order.

### 2) V16 classifier health + controls

- `src/api/v1/health.py`
  - `GET /api/v1/health/v16-classifier` (and alias `/api/v1/v16-classifier/health`)
    - Reports: `loaded`, `speed_mode`, cache stats, DWG converter availability, categories.
    - Uses `PartClassifierV16.get_cache_stats()` / internal fields when available (keeps accurate metrics).
  - `POST /api/v1/health/v16-classifier/cache/clear` (admin token required)
    - Clears cache via `PartClassifierV16.clear_cache()` (or best-effort fallback).
  - `POST /api/v1/health/v16-classifier/speed-mode` (admin token required)
    - Updates `classifier.speed_mode` and internal speed config (no torch import needed).
  - `GET /api/v1/health/v16-classifier/speed-mode`
    - Returns current mode (or `unavailable` if classifier not loaded).

## Verification

```bash
.venv/bin/pytest -q tests/integration/test_analyze_batch_classify_api.py
.venv/bin/pytest -q tests/unit/test_health_v16_classifier_endpoints.py
```

Result:

- `test_analyze_batch_classify_api.py`: `1 passed`
- `test_health_v16_classifier_endpoints.py`: `5 passed`

## Notes / Limits

- In environments without `torch` (common for lightweight dev/CI), V16/V6 ML classifiers may be unavailable; the endpoints are designed to degrade gracefully and still return structured responses.

