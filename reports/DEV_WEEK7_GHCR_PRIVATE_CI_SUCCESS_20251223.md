# DEV_WEEK7_GHCR_PRIVATE_CI_SUCCESS_20251223

## Context
- GHCR package `zensgit/dedupcad-vision` remains private.
- CI relies on `GHCR_TOKEN` to authenticate image pulls.

## Verification
- Package visibility: `gh api /user/packages/container/dedupcad-vision` -> `private`.
- Unauthenticated manifest fetch returns `401`.
- Repo secret check: `gh secret list -R zensgit/cad-ml-platform` shows `GHCR_TOKEN`.
- CI run success: https://github.com/zensgit/cad-ml-platform/actions/runs/20462715500

## Tests
- `pytest tests/unit/test_recovery_state_redis_roundtrip.py -q`

## Results
- Local unit test passed.
- CI is green with authenticated GHCR pulls.

## Notes
- Keep `GHCR_TOKEN` until the package is explicitly made public.
