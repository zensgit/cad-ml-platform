# Integration Test Report (DedupCAD Vision)

- Date: 2025-12-28
- Scope: `tests/integration` with live dedupcad-vision

## Setup
- dedupcad-vision server: `python start_server.py --port 58001`
- Env:
  - `DEDUPCAD_VISION_URL=http://localhost:58001`
  - `DEDUPCAD_VISION_REQUIRED=1`

## Command
- `DEDUPCAD_VISION_URL=http://localhost:58001 DEDUPCAD_VISION_REQUIRED=1 .venv/bin/python -m pytest tests/integration -q`

## Result
- PASS

## Summary
- Tests: 24 passed, 4 skipped
- Duration: 3.58s
