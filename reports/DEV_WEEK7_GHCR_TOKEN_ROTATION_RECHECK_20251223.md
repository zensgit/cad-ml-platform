# DEV_WEEK7_GHCR_TOKEN_ROTATION_RECHECK_20251223

## Context
- User indicated GHCR_TOKEN rotated again (read-only PAT).
- Goal: validate secret update and CI pull of private GHCR image.

## Verification
- `gh secret list -R zensgit/cad-ml-platform` still shows GHCR_TOKEN updated_at `2025-12-23T14:27:34Z`.
  - This timestamp did not change from the prior rotation check, so the new rotation could not be confirmed via API.

## CI Attempts
- Run: https://github.com/zensgit/cad-ml-platform/actions/runs/20463631075 (attempts 1 and 2)
- Outcome: `lint-type` and `lint-all-report` failed immediately; `tests` and `e2e-smoke` skipped.
- Job metadata shows no runner assignment (runner_name empty) and no logs available via API (`/actions/jobs/{id}/logs` returns 404).

## Tests
- `pytest tests/unit/test_recovery_state_redis_roundtrip.py -q`

## Results
- Local unit test passed.
- CI failure appears unrelated to GHCR and occurred before runner assignment; logs unavailable.

## Follow-up
- Re-run `gh secret set GHCR_TOKEN -R zensgit/cad-ml-platform` to confirm the updated timestamp changes.
- Re-trigger CI after confirming the secret update (or retry if GitHub runner capacity was temporarily unavailable).
