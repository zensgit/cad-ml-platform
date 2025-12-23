# Unit Test Fixes (Week 7)

## Summary
- Fixed dedup2d worker storage creation to be patchable in tests.
- Converted OCR distributed control tests to async to avoid missing event loop in Py3.11+.
- Added compatibility `get_client()` wrapper for FAISS recovery state tests.
- Updated vector migration metrics test monkeypatch to match new `upgrade_vector` signature.

## Code Changes
- `src/core/dedupcad_2d_worker.py`
  - Use module import for `create_dedup2d_file_storage()` to allow test patching.
- `tests/ocr/test_distributed_control.py`
  - Converted tests to `@pytest.mark.asyncio` and removed manual loop usage.
- `src/core/similarity.py`
  - Added `get_client()` wrapper and used it for recovery state storage.
- `tests/unit/test_vector_migrate_metrics.py`
  - Updated monkeypatched `upgrade_vector` signature.

## Tests
- `pytest tests/unit/test_dedup_2d_jobs_redis.py -q`
  - Result: PASS (23 passed)
- `pytest tests/ocr/test_distributed_control.py -q`
  - Result: PASS (2 passed)
- `pytest tests/unit/test_recovery_state_redis_roundtrip.py -q`
  - Result: PASS (1 passed)
- `pytest tests/unit/test_vector_migrate_api.py tests/unit/test_vector_migrate_v4.py tests/unit/test_vector_migrate_metrics.py -q`
  - Result: PASS (9 passed)
- `pytest tests/unit -q`
  - Result: PASS (3328 passed, 32 skipped)
  - Warnings:
    - Unknown pytest mark `slow` (2 warnings)
    - Pydantic v2 deprecation warnings in adapter factory tests (2 warnings)
    - Unraisable exception warning from `TelemetryIngestor._run` when loop closes

## Notes
- Unit suite passes locally under `.venv` Python 3.11.
- The remaining warnings are non-fatal but can be addressed if you want a clean run.
