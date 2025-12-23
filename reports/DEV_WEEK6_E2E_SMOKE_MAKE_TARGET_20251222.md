# Week6 Add E2E Smoke Make Target (2025-12-22)

## Change
- Added `make e2e-smoke` target to run E2E regression + dedupcad-vision contract tests.
- Documented the target in README with env var hints.

## Test
- `make e2e-smoke`

## Result
- `4 passed in 2.43s`

## Notes
- Defaults to `API_BASE_URL=http://localhost:8000`, `DEDUPCAD_VISION_URL=http://localhost:58001`.
