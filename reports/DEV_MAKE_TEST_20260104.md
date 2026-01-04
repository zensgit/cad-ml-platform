# make test Report

- Date: 2026-01-04
- Scope: `make test`

## Command
- make test

## Result
- PASS

## Summary
- 3993 passed, 21 skipped, 3 warnings
- Coverage: 71% (htmlcov generated)
- Duration: 153.86s

## Update (warnings enabled)
### Command
- PYTHONWARNINGS=default .venv/bin/python -m pytest tests -v --cov=src --cov-report=term-missing --cov-report=html -W default

### Result
- 3993 passed, 21 skipped, 170 warnings
- Coverage: 71% (htmlcov generated)
- Duration: 105.37s

### Notes
- Warnings summary dominated by ResourceWarning: unclosed event loop.
