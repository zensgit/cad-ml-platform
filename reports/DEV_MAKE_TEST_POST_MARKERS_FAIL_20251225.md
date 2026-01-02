# make test (post markers + pydantic fix) - failed

- Date: 2025-12-25
- Command: `make test`
- Result: FAIL
- Failures:
  - `tests/unit/test_batch_similarity_faiss_unavailable.py::test_batch_similarity_faiss_unavailable_degraded_flag`
  - `tests/unit/test_batch_similarity_faiss_unavailable.py::test_batch_similarity_degraded_forces_fallback`
- Failure detail: response `fallback` was `null` (expected `true`).
- Coverage: 71% total, HTML output in `htmlcov/`.
