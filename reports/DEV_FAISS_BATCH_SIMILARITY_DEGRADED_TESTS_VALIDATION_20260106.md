# DEV_FAISS_BATCH_SIMILARITY_DEGRADED_TESTS_VALIDATION_20260106

## Scope
Validate Faiss batch similarity degraded fallback tests and metrics handling.

## Command
- `pytest tests/unit/test_faiss_degraded_batch.py -v`

## Results
- 8 passed, 1 skipped (`prometheus_client` not available for metric delta check).
