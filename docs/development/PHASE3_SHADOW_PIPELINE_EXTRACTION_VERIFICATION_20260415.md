# Phase 3 Shadow Pipeline Extraction Verification

## Summary
- Extracted shadow evidence collection from `src/api/v1/analyze.py` into `src/core/classification/shadow_pipeline.py`.
- Kept `FusionAnalyzer` and `Hybrid` override ordering in `analyze.py`.
- Preserved analyze-level access to `_enrich_graph2d_prediction`, `_build_graph2d_soft_override_suggestion`, `_resolve_history_sequence_file_path`, and `_safe_float_env`.

## Files Changed
- `src/core/classification/shadow_pipeline.py`
- `src/core/classification/__init__.py`
- `src/api/v1/analyze.py`
- `tests/unit/test_classification_shadow_pipeline.py`
- `tests/integration/test_analyze_dxf_fusion.py`
- `docs/development/PHASE3_SHADOW_PIPELINE_EXTRACTION_DEVELOPMENT_PLAN_20260415.md`
- `docs/development/PHASE3_SHADOW_PIPELINE_EXTRACTION_VERIFICATION_20260415.md`

## Validation
- `python3 -m py_compile src/core/classification/shadow_pipeline.py src/core/classification/__init__.py src/api/v1/analyze.py tests/unit/test_classification_shadow_pipeline.py tests/integration/test_analyze_dxf_fusion.py`
- `.venv311/bin/flake8 src/core/classification/shadow_pipeline.py src/core/classification/__init__.py src/api/v1/analyze.py tests/unit/test_classification_shadow_pipeline.py tests/integration/test_analyze_dxf_fusion.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_classification_shadow_pipeline.py tests/unit/test_analyze_graph2d_gate_helpers.py tests/unit/test_analyze_history_sequence_resolution.py tests/unit/test_classification_baseline_policy.py tests/unit/test_classification_override_policy.py tests/unit/test_classification_finalization.py tests/unit/test_classification_decision_contract.py`
  - Result: `42 passed, 7 warnings`
- `.venv311/bin/python -m pytest -q tests/integration/test_analyze_dxf_fusion.py tests/integration/test_analyze_dxf_graph2d_prediction_contract.py`
  - Result: `15 passed, 7 warnings`

## Assertions Locked By This Batch
- Graph2D gate enrichment and soft override helpers still behave the same through `src.api.v1.analyze`.
- History sidecar resolution still produces the same `history_sequence_input`.
- Hybrid shadow mapping still populates `fine_*`, `hybrid_decision`, and rejection metadata.
- Part classifier shadow flow still normalizes `part_family` fields.
- `fusion_inputs["l4"]` still prefers Graph2D over ML when Graph2D fusion is enabled.
