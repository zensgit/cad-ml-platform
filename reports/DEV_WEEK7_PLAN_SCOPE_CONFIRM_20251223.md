# DEV_WEEK7_PLAN_SCOPE_CONFIRM_20251223

## Scope
- Run local full `make test` with current pytest ignores.
- Validate ignored tests via `pytest --override-ini addopts=`.
- If stable, remove ignores from `pytest.ini` and re-run `make test`.

## Tests
- `pytest tests/unit/test_recovery_state_redis_roundtrip.py -q`

## Results
- Scope confirmed.
- Sanity test passed.
