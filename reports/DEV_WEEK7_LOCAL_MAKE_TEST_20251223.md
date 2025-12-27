# DEV_WEEK7_LOCAL_MAKE_TEST_20251223

## Scope
- Run local full `make test` with current pytest ignores.

## Attempts
1) `make test`
   - Failed during collection on Python 3.9 (PEP604 `|` annotations).
2) `make PYTHON=.venv/bin/python test`
   - Passed on Python 3.11.13.

## Results
- `3648 passed, 42 skipped, 5 warnings`.
- Coverage HTML written to `htmlcov/`.

## Notes
- Project requires Python 3.10+; use `.venv/bin/python` for local full test runs.
