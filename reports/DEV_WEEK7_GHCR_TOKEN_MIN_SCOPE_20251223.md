# DEV_WEEK7_GHCR_TOKEN_MIN_SCOPE_20251223

## Context
- GHCR package remains private, CI uses `GHCR_TOKEN`.
- Goal: keep token permissions minimal (read-only for packages).

## Changes
- Documented GHCR private usage and minimal token scope in `README.md`.

## Validation
- `pytest tests/unit/test_recovery_state_redis_roundtrip.py -q`

## Results
- Test passed.

## Pending
- GHCR_TOKEN rotation to a read-only PAT requires a new token value.
- Suggested command once you create a PAT with `read:packages`:
  `gh secret set GHCR_TOKEN -R zensgit/cad-ml-platform --body '<NEW_PAT>'`
