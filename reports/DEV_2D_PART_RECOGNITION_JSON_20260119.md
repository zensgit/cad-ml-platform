# DEV_2D_PART_RECOGNITION_JSON_20260119

## Summary
- Added JSON adapter for v2 geometry payloads and enriched DXF parsing with text/dimension metadata to support 2D part recognition signals.
- Built text signals (filename + extracted text) and enabled L2 fusion classification when 3D features are absent.
- Expanded L2 feature inputs to include dimension and text counts, and documented JSON input usage with a sample payload.
- Added an integration test for DXF keyword-driven fusion classification.
- Added a negative integration test to ensure JSON uploads without keywords fall back to rule-based classification.
- Allowed application/json MIME and JSON file format in the analysis pipeline.

## Verification
- `pytest tests/unit/test_adapter_factory_coverage.py -v`
- `pytest tests/unit/test_input_validator_coverage.py -v`
- `pytest tests/integration/test_analyze_json_fusion.py -v`
- `pytest tests/integration/test_analyze_dxf_fusion.py -v`
- `pytest tests/integration/test_e2e_api_smoke.py -v` (1 passed, 1 skipped: dedup vision unavailable)
