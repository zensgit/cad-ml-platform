# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_LABEL_OVERRIDE_20260122

## Summary
- Added Graph2D label override logic for high-confidence labels (模板/零件图/装配图).
- Introduced configurable env controls for override labels and threshold.
- Updated fusion schema to v1.2 and annotated L4 sources.

## Code Changes
- `src/core/knowledge/fusion_analyzer.py` (override logic, graph2d source handling)
- `src/core/knowledge/fusion_contracts.py` (schema v1.2)
- `src/api/v1/analyze.py` (l4_prediction source tagging)
- `.env.example` (override env settings)
- `src/main.py` (env validation)
- `tests/integration/test_analyze_dxf_fusion.py` (schema version)

## Env Settings
- `FUSION_GRAPH2D_OVERRIDE_LABELS=模板,零件图,装配图`
- `FUSION_GRAPH2D_OVERRIDE_MIN_CONF=0.6`

## Notes
- Override applies only when Graph2D is the L4 source and no high conflict is detected.
