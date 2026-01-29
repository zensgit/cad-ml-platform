# DEV_MATERIAL_API_FIXES_20260129

## Goal
Fix material API semantics and improve material-aware process routing behavior based on recent repository changes.

## Changes Implemented
- Material search now returns the real `match_type` (exact/grade/name/alias/pinyin/etc.) and respects `category`/`group` filters even on exact matches.
- Galvanic corrosion risk levels aligned to `moderate` (instead of `medium`) to match tests and API expectations.
- Heat-treatment compatibility returns `allowed` instead of `neutral` when not explicitly recommended/forbidden.
- Cost search supports an `include_estimated` switch to include group-based cost estimates.
- Cost comparison now returns a `missing` list for unknown grades.
- Updated README materials API notes and CHANGELOG entry.
- Added release/PR summary report for the materials API change set.
- Material classification in process routing avoids false positives for short tokens by requiring boundaries for short ASCII patterns.
- Added OCR endpoint unit coverage for `material_info` when title-block material is present.

## Files Updated
- `src/core/materials/classifier.py`
- `src/api/v1/materials.py`
- `src/core/process/route_generator.py`
- `tests/unit/test_ocr_endpoint_coverage.py`

## Notes
These changes are backwards compatible with existing API responses and add better semantic clarity to search and compatibility endpoints.
