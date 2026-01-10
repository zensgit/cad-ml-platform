# DedupCAD Vision Full Integration Re-Run

- Date: 2025-12-28
- Scope: `tests/integration` with live `dedupcad-vision`

## Environment
- `dedupcad-vision` started via `python start_server.py` (port 58001)
- `DEDUPCAD_VISION_REQUIRED=1`
- `PYTHONPATH=/Users/huazhou/Downloads/Github/cad-ml-platform`

## Commands
- `pytest tests/integration -q`

## Results
- PASS (24 passed, 4 skipped)

## Notes
- First run failed with `ModuleNotFoundError: jose` due to missing `PYTHONPATH` in the test environment; re-run with `PYTHONPATH` succeeded.
- Service stopped after tests.
