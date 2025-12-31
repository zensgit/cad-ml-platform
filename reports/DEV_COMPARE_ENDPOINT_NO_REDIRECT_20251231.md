# Compare Endpoint No-Redirect Check (2025-12-31)

## Scope

- Avoid 307 redirect on `POST /api/compare` by aligning route path.
- Confirm endpoint responds without redirect.

## Changes

- `src/api/v1/compare.py`: switch route path to `""` to bind `/api/compare` directly.
- `tests/unit/test_compare_endpoint.py`: add `follow_redirects=False` assertion.

## Tests

```bash
.venv/bin/python -m pytest tests/unit/test_compare_endpoint.py -v
```

Result:
- `3 passed in 2.13s`
