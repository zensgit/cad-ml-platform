# DEV_DXF_FILENAME_HYBRID_INTEGRATION_20260125

## Objective
Integrate filename-based classification and hybrid fusion into the DXF analysis pipeline with safe feature flags and batch output visibility.

## Implementation
- `src/api/v1/analyze.py`
  - Added `filename_prediction` and `hybrid_decision` to classification payload.
  - Added optional hybrid override via `HYBRID_CLASSIFIER_OVERRIDE` and `HYBRID_OVERRIDE_MIN_CONF`.
- `scripts/batch_analyze_dxf_local.py`
  - Added CSV columns for filename and hybrid outputs:
    - `filename_label`, `filename_confidence`, `filename_match_type`, `filename_extracted_name`
    - `hybrid_label`, `hybrid_confidence`, `hybrid_source`, `hybrid_path`

## Feature Flags
- `HYBRID_CLASSIFIER_ENABLED` (default: true)
- `FILENAME_CLASSIFIER_ENABLED` (default: true)
- `FILENAME_MIN_CONF` (default: 0.8)
- `FILENAME_FUSION_WEIGHT` (default: 0.7)
- `GRAPH2D_FUSION_WEIGHT` (default: 0.3)
- `HYBRID_CLASSIFIER_OVERRIDE` (default: false)
- `HYBRID_OVERRIDE_MIN_CONF` (default: 0.8)

## Notes
- Hybrid logic uses filename-high-confidence adoption first, then Graph2D if filename is low, then weighted fusion.
- Override is off by default to support safe rollout and easy rollback.
