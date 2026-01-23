# DEV_MECH_KNOWLEDGE_4000CAD_FUSION_ANALYZER_UPGRADE_VALIDATION_20260121

## Checks
- FusionAnalyzer schema bumped to v1.1 with normalized feature hashing.
- Startup env validation emits warnings for misconfigured fusion/graph2d flags.
- Integration test validates fusion_inputs payload when analyzer is enabled.

## Tests
- `pytest tests/integration/test_analyze_dxf_fusion.py -v`

## Files Verified
- `src/core/knowledge/fusion_contracts.py`
- `src/core/knowledge/fusion_analyzer.py`
- `src/main.py`
- `tests/integration/test_analyze_dxf_fusion.py`
