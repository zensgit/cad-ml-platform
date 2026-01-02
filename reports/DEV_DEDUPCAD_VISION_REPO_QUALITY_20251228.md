# DedupCAD Vision Repo Quality Report

- Date: 2025-12-28
- Scope: `dedupcad-vision` repo (`/Users/huazhou/Downloads/Github/dedupcad-vision`)

## Commands
- `ruff check .`
- `pytest -q`
- `mypy src`

## Results
- Ruff: PASS
- Pytest: PASS (2058 passed, 23 skipped, 125 warnings)
- Mypy: PASS (notes about annotation-unchecked in health checks)

## Notes
- Pytest warnings are from `ezdxf`/`pyparsing` deprecations.
- Mypy notes: `annotation-unchecked` for untyped functions in `src/caddedup_vision/monitoring/health.py`.
