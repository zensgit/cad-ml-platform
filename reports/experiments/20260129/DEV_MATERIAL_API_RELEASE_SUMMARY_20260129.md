# DEV_MATERIAL_API_RELEASE_SUMMARY_20260129

## Release/PR Summary
- Materials API: cost search adds `include_estimated` (group-based estimates) and cost compare returns `missing` list for unknown grades.
- Compatibility semantics aligned (`moderate` galvanic risk, `allowed` heat-treatment fallback).
- Search results now surface accurate `match_type` and respect `category/group` filters on exact matches.
- OCR response includes `material_info` with process warnings/recommendations.

## CI/Dev Setup Note
CI workflows already install `requirements-dev.txt`; no workflow changes needed. README now clarifies pytest-asyncio is included for async OCR tests.

## Validation
Targeted unit tests:
- `tests/unit/test_materials_api.py`
- `tests/unit/test_material_classifier.py`
- `tests/unit/test_ocr_endpoint_coverage.py`
- `tests/unit/test_route_generator.py`

Result: 1627 passed.
