# Features Diff Qdrant Validation 2026-03-07

## Scope
- Route `GET /api/v1/features/diff`
- Qdrant backend compatibility when `VECTOR_STORE_BACKEND=qdrant`

## Changes
- Added a local Qdrant store resolver in `src/api/v1/features.py`.
- `features/diff` now loads both vectors from Qdrant when the backend is `qdrant`.
- Preserved the legacy in-memory path as fallback.
- Reused stored payload metadata to resolve `feature_version` under the Qdrant path.

## Validation Commands
```bash
python3 -m py_compile \
  src/api/v1/features.py \
  tests/unit/test_features_diff_endpoint.py

flake8 \
  src/api/v1/features.py \
  tests/unit/test_features_diff_endpoint.py \
  --max-line-length=100

pytest -q tests/unit/test_features_diff_endpoint.py
```

## Coverage
- In-memory diff path still works
- Qdrant success path
- Qdrant not-found path
- Qdrant dimension-mismatch path
