# make test (post fallback fix)

- Date: 2025-12-25
- Command: `make test`
- Result: PASS (3949 passed, 29 skipped, 5 warnings in 93.98s)
- Coverage: 71% total, HTML output in `htmlcov/`

Warnings:
- Unknown pytest marks: `performance`, `slow`.
- Pydantic deprecated `__fields__` usage warning during tests.

Notes:
- `make` warned about overriding `security-audit` target; test run completed normally.
