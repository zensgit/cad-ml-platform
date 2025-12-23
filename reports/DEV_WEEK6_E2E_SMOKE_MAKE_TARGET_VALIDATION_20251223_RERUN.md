# Week6 E2E Smoke - Make Target Validation (Rerun 2025-12-23)

## Change
- Updated `tests/conftest.py` to avoid deprecated `asyncio.get_event_loop()` access in sync fixtures.

## Command
- `make e2e-smoke`

## Environment
- API_BASE_URL: default (`http://localhost:8000`)
- DEDUPCAD_VISION_URL: default (`http://localhost:58001`)

## Result
- Passed: 4
- Failed: 0
- Duration: 3.08s

## Warnings
- None
