# DEV_CI_BATCH_SIMILARITY_FAISS_FALLBACK_20251229

## Scope
- Re-run CI after batch similarity fallback stabilization.

## Changes
- `src/core/similarity.py`
  - Attach backend metadata to vector stores.
- `src/api/v1/vectors.py`
  - Use store metadata to detect fallback.
- `tests/unit/test_batch_similarity_faiss_unavailable.py`
  - Patch `get_vector_store` to enforce fallback metadata during the test.

## Validation
- Command: `gh workflow run "CI" -r feat/dedup2d-phase4-production-ready`
  - Run: `20558653314` (https://github.com/zensgit/cad-ml-platform/actions/runs/20558653314)
  - Result: success.
  - Jobs: `lint-type`, `lint-all-report`, `tests (3.10)`, `tests (3.11)`, `e2e-smoke` all completed.
  - Note: CI still emits annotation `lint-type: .github#29464` but job succeeds.
