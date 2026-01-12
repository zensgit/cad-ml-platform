# DEV_DRAWING_RECOGNITION_FULL_TESTS_20260112

## Scope
Attempt full test suite for the drawing recognition changes.

## Command
```bash
pytest tests -v --cov=src --cov-report=term-missing
```

## Result
- Test collection failed due to missing dependency:
  - `ModuleNotFoundError: No module named 'jwt'` while importing `tests/unit/test_integration_auth_middleware.py`.

## Notes
- Install `PyJWT` (or ensure `jwt` is available) before rerunning full test coverage.
