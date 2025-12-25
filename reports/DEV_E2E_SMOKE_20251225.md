# DEV_E2E_SMOKE_20251225

## Scope

- Run `make e2e-smoke` against local services.

## Validation

- Command: `make e2e-smoke`
- Result: 1 passed, 3 skipped
  - Passed: core API smoke
  - Skipped: dedup search (503), dedupcad-vision contract tests (service unreachable)

## Notes

- `cad-ml-api` container was running during the test.
- `make` printed warnings about duplicate `security-audit` target (pre-existing).
