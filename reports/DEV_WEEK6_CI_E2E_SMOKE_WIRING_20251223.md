# Week6 CI Wiring - E2E Smoke (2025-12-23)

## Change
- Added `e2e-smoke` job to `.github/workflows/ci.yml` (stub-based).
- Documented `make e2e-smoke` in CI optimization summary.

## Test
- `make e2e-smoke`

## Result
- Passed: 4
- Failed: 0

## Notes
- CI starts a local `dedupcad-vision` stub and waits for `/health` before running tests.
