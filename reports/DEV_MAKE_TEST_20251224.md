# DEV_MAKE_TEST_20251224

## Scope
- Full regression via `make test` using project venv Python.

## Validation
- Command: `make PYTHON=.venv/bin/python test`
  - Result: 3950 passed, 28 skipped, 5 warnings.
  - Coverage: 71% (htmlcov generated).

## Notes
- Makefile emits duplicate target warnings for `security-audit` (existing).
