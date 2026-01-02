# Dedup2D Async Precision Overlay Test (2026-01-01)

## Scope

- Add async job coverage with precision overlay enabled.
- Ensure test harness tolerates missing PyJWT dependency.

## Changes

- Added async precision overlay test in `tests/test_dedup_2d_proxy.py`.
- Guarded integration auth middleware to avoid import errors when PyJWT is absent.

## Tests

- `pytest tests/test_dedup_2d_proxy.py -k "async_with_precision_overlay" -v`

## Results

- OK: 1 passed (28 deselected).
