# Render + feedback tests (2025-12-31)

## Scope
- Added API tests for the render and feedback endpoints.
- Render tests stub the dedupcad worker import to avoid `arq` dependency in unit tests.

## Command
- `.venv/bin/python -m pytest tests/test_render.py tests/test_feedback.py -v`

## Results
- **Pass**: 5
- **Fail**: 0
- **Duration**: 4.31s

## Notes
- Initial `python -m pytest` failed because `python` was not on PATH; reran with venv Python.
