# DEV_DXF_FILENAME_HYBRID_INTEGRATION_VALIDATION_20260125

## Validation Summary
- Confirmed hybrid/filename outputs are available in API payload and batch CSV.
- Added unit tests for filename extraction and hybrid decision logic.

## Checks
- Unit tests: `pytest tests/unit/test_filename_classifier.py -v`
  - 5 passed
- Batch output fields in `scripts/batch_analyze_dxf_local.py`:
  - `filename_label`, `filename_confidence`, `filename_match_type`, `filename_extracted_name`
  - `hybrid_label`, `hybrid_confidence`, `hybrid_source`, `hybrid_path`
