# DEV_MECH_KNOWLEDGE_4000CAD_FUSION_ANALYZER_UPGRADE_20260121

## Summary
- Expanded FusionAnalyzer normalization inputs across L2/L3 and added confidence calibration heuristics.
- Bumped fusion schema version to v1.1 and attached feature vector hashes for traceability.
- Added startup validation for fusion/graph2d env flags and extended integration test coverage for fusion inputs.

## Code Changes
- `src/core/knowledge/fusion_contracts.py` (schema v1.1, normalization schema update)
- `src/core/knowledge/fusion_analyzer.py` (normalization, calibration, feature hash)
- `src/main.py` (env flag validation on startup)
- `tests/integration/test_analyze_dxf_fusion.py` (fusion_inputs assertions)

## Notes
- Normalization now includes L2 geometry counts and L3 B-Rep metrics + surface type counts.
- AI confidence is calibrated with conflict penalties and low-signal downweighting.
