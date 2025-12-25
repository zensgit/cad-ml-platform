# DEV_MAKE_TEST_20251225

## Scope

- Run full `make test` using the project Python environment.
- Ensure batch similarity reports degraded state when falling back to memory.

## Changes

- Prefer `.venv/bin/python` (or python3.11/3.10) in `Makefile` to align with 3.10+.
- Treat fallback as degraded in `/api/v1/vectors/similarity/batch` responses.

## Validation

- Command: `make test`
- Result: passed (3950 passed, 28 skipped, 5 warnings) in ~1m40s; coverage 71% (HTML in `htmlcov/`).
- Command: `.venv/bin/python -m pytest tests/unit/test_batch_similarity_faiss_unavailable.py::test_batch_similarity_faiss_unavailable_degraded_flag -vv`
- Result: passed

## Notes

- `make` warns about duplicate `security-audit` target (pre-existing).
