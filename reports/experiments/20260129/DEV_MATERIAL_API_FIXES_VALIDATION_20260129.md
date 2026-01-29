# DEV_MATERIAL_API_FIXES_VALIDATION_20260129

## Validation Scope
- Material search semantics and compatibility status changes
- Cost search parameterization and cost-compare missing list
- Process route material pattern matching behavior
- OCR `material_info` unit coverage

## Test Command
```bash
.venv-graph/bin/python -m pip install pytest-asyncio
.venv-graph/bin/python -m pytest \
  tests/unit/test_materials_api.py \
  tests/unit/test_material_classifier.py \
  tests/unit/test_ocr_endpoint_coverage.py \
  tests/unit/test_route_generator.py -q
```

## Results
- Passed: 1627
- Failed: 0
- Duration: 11.96s

## Follow-up
None.

## Artifacts
- This report: `reports/experiments/20260129/DEV_MATERIAL_API_FIXES_VALIDATION_20260129.md`
