# Full Test Run (DedupCAD Vision Required)

- Date: 2025-12-28
- Scope: `make test` with live `dedupcad-vision`

## Environment
- `dedupcad-vision` started via `python start_server.py` (port 58001)
- `DEDUPCAD_VISION_REQUIRED=1`
- `PYTHONPATH=/Users/huazhou/Downloads/Github/cad-ml-platform`

## Command
- `make test`

## Results
- PASS (3956 passed, 25 skipped)
- Coverage: 71% (htmlcov generated)

## Notes
- `dedupcad-vision` stopped after tests.
