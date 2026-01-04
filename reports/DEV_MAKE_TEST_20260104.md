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

## Update (ResourceWarning as error)
### Command
- PYTHONWARNINGS=default .venv/bin/python -m pytest tests -v --cov=src --cov-report=term-missing --cov-report=html -W error::ResourceWarning

### Result
- 3993 passed, 21 skipped, 158 warnings
- Coverage: 71% (htmlcov generated)
- Duration: 104.37s

### Notes
- PytestUnraisableExceptionWarning surfaced unclosed socket ResourceWarnings.

## Update (asyncio debug)
### Command
- PYTHONASYNCIODEBUG=1 PYTHONWARNINGS=default .venv/bin/python -m pytest tests -v --cov=src --cov-report=term-missing --cov-report=html -W default

### Result
- 3993 passed, 21 skipped, 180 warnings
- Coverage: 72% (htmlcov generated)
- Duration: 110.02s

### Notes
- ResourceWarning for unclosed event loop remains the dominant warning.
