# DEV_DXF_TITLEBLOCK_FEATURES_VALIDATION_20260125

## Validation Summary
- Verified title-block extraction and classifier mapping via unit tests.

## Tests
```
.venv-graph/bin/python -m pytest tests/unit/test_titleblock_extractor.py -v
```
- 3 passed

## Checks
- `src/api/v1/analyze.py` exposes `titleblock_prediction`.
- `scripts/batch_analyze_dxf_local.py` emits title-block fields in CSV output.
- Attribute-driven title block extraction validated via unit test.
