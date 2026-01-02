# DedupCAD Vision Compare Endpoint Compatibility (2025-12-31)

## Scope

- Add `/api/compare` fallback endpoint for dedupcad-vision L3 feature comparison.
- Provide `/api/v1/compare` alias and update contract documentation.

## Changes

- Added `src/api/v1/compare.py` to compute cosine similarity against stored vectors.
- Registered router in `src/api/__init__.py` (exposes `/api/compare` + `/api/v1/compare`).
- Updated `docs/DEDUP2D_VISION_INTEGRATION_CONTRACT.md` with request/response spec.
- Added unit tests for success, missing candidate, and dimension mismatch paths.

## Tests

```bash
.venv/bin/python -m pytest tests/unit/test_compare_endpoint.py -v
```

Result:
- `3 passed in 2.39s`
