# DEV_E2E_SMOKE_VISION_REQUIRED_20251225_B

## Scope

- Start dedupcad-vision locally and rerun E2E smoke with vision required after reset.

## Validation

- Command: `DEDUPCAD_VISION_REQUIRED=1 make e2e-smoke`
- Result: 4 passed (38.57s)

## Notes

- `make` printed warnings about duplicate `security-audit` target (pre-existing).
