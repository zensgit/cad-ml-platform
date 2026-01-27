# DEV_TRAINING_DXF_SOFT_OVERRIDE_SUGGESTION_OUTPUT_20260123

## Summary
- Added a non-blocking `soft_override_suggestion` field to `/api/v1/analyze` classification output.
- Updated batch DXF analysis script to include soft-override columns and summary count.

## Implementation
- API: `src/api/v1/analyze.py`
  - Emits `soft_override_suggestion` when Graph2D prediction is available.
  - Default threshold from `GRAPH2D_SOFT_OVERRIDE_MIN_CONF` (default 0.17).
  - Eligibility requires: rule_version=v1, confidence_source=rules, Graph2D allowed/not excluded, and confidence >= threshold.
- Batch script: `scripts/batch_analyze_dxf_local.py`
  - Outputs columns: soft_override_eligible/label/confidence/threshold/reason.
  - Adds `soft_override_candidates` to summary.json.

## Notes
- This is a suggestion-only field; it does not alter `part_type` or `confidence`.
- Use env `GRAPH2D_SOFT_OVERRIDE_MIN_CONF` to adjust the suggestion threshold.
