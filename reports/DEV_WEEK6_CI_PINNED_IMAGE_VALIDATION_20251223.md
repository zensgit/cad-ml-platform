# Week6 CI - Pinned Image Validation (2025-12-23)

## Scope
- Validate CI wiring changes for pinned dedupcad-vision image.
- Validate E2E smoke vector probe normalization.

## Tests
- `pytest tests/integration/test_e2e_api_smoke.py -q`
- `pytest tests/integration/test_dedupcad_vision_contract.py -q`

## Results
- Passed: 4
- Failed: 0

## Warnings
- `DeprecationWarning: There is no current event loop` from `tests/conftest.py:164` (present in both runs).

## Notes
- E2E smoke and vision contract tests pass locally with default settings.
