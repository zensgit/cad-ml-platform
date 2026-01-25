# DEV_DXF_TITLEBLOCK_REFINEMENTS_20260125

## Objective
Improve title-block signal extraction for DXF drawings by capturing block attributes, normalizing escaped text, and gating title-block overrides.

## Implementation
- `src/ml/titleblock_extractor.py`
  - Added normalized text handling for escaped sequences.
  - Decodes MIF-encoded `\\M+` sequences to restore title-block tags.
  - Added extraction from INSERT block attributes (ATTRIB) with tag-aware parsing.
  - Tag hints now populate part name, drawing number, material, scale, and revision.
- `src/ml/hybrid_classifier.py`
  - Added `TITLEBLOCK_OVERRIDE_ENABLED` to disable direct title-block overrides by default.
- `tests/unit/test_titleblock_extractor.py`
  - Added attribute-based extraction coverage.

## Supporting Data
- `reports/experiments/20260123/titleblock_raw_text_samples_20260125.csv`
  - Sampled raw title-block texts from DXF files to verify source fields.

## Notes
- Tag-aware extraction allows title-block labels even when raw text lacks explicit "名称" patterns.
- Override behavior remains opt-in via `TITLEBLOCK_OVERRIDE_ENABLED=true`.
