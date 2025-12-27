# DEV_WEEK7_MAKE_TEST_POST_UNIGNORE_20251223

## Scope
- Remove pytest ignore list from `pytest.ini`.
- Run full `make test` with coverage using project venv Python.

## Changes
- Removed temporary ignores in `pytest.ini` for phase14/18/19/22 and vision API integration tests.

## Tests
- Command: `make PYTHON=.venv/bin/python test`
- Result: `3934 passed, 42 skipped, 5 warnings`.
- Coverage: `71%` total (HTML report written to `htmlcov`).

## Warnings Observed
- Makefile target override warning: `security-audit` target redefined.
- Unknown pytest marks: `performance`, `slow`.
- Pydantic v2 deprecation warning for `__fields__`.

## Files Updated
- `pytest.ini`
