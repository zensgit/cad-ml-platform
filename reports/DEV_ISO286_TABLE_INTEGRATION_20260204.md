# DEV_ISO286_TABLE_INTEGRATION_20260204

## Summary
- Added ISO 286 deviation table loading (`iso286_deviations.json`) with cached lookups and a new `get_limit_deviations` helper.
- Updated assistant tolerance retrieval to prefer ISO 286 table bounds when symbol + grade + size are provided, with fallback to IT-grade logic.
- Replaced `print` error paths in knowledge retrieval with structured logging and cached precision rule loading.
- Updated tolerance knowledge design doc to reflect table-backed deviations and new API.
- Added unit tests covering hole/shaft limit deviations (H7, h6, g6, K7, P7, JS6).

## Verification
- `python3 -m pytest tests/unit/test_tolerance_limit_deviations.py -q`
- `python3 -m pytest tests/unit/test_tolerance_fundamental_deviation.py -q`

## Notes
- Table-backed deviations are optional and loaded from `ISO286_DEVIATIONS_PATH` (default: `data/knowledge/iso286_deviations.json`).
- When table data is missing, tolerance lookup falls back to IT-grade + fundamental deviation logic.
