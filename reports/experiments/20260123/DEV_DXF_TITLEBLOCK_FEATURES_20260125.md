# DEV_DXF_TITLEBLOCK_FEATURES_20260125

## Objective
Extract title-block text signals from DXF drawings and expose them through the hybrid classifier for fusion decisions.

## Implementation
- `src/ml/titleblock_extractor.py`
  - Added synonym loading (default: `data/knowledge/label_synonyms_template.json`).
  - Added singleton accessor for `TitleBlockClassifier`.
  - Extracts title-block values from block attributes (ATTRIB) and normalizes escaped text.
- `src/ml/hybrid_classifier.py`
  - Added title-block branch with configurable weight and min confidence.
  - Added `TITLEBLOCK_OVERRIDE_ENABLED` to control direct title-block overrides.
  - Included title-block prediction in fusion decision path.
- `src/api/v1/analyze.py`
  - Exposed `titleblock_prediction` in classification payload.
- `scripts/batch_analyze_dxf_local.py`
  - Added title-block fields to batch CSV output.

## Feature Flags
- `TITLEBLOCK_ENABLED` (default: false)
- `TITLEBLOCK_MIN_CONF` (default: 0.75)
- `TITLEBLOCK_FUSION_WEIGHT` (default: 0.2)
- `TITLEBLOCK_REGION_X_RATIO` / `TITLEBLOCK_REGION_Y_RATIO`
- `TITLEBLOCK_OVERRIDE_ENABLED` (default: false)

## Notes
- Title-block classification is only attempted when `TITLEBLOCK_ENABLED=true` and DXF bytes are available.
- Fusion adds a bonus when multiple sources agree on the same label.
