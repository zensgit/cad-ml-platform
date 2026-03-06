# AI Parallel Delivery Validation 2026-03-06

## Summary

This delivery completed the first integrated pass of the parallel AI roadmap on top of
the current `main` baseline:

- output normalization for coarse/fine DXF labels
- lightweight knowledge productization in the analyze response
- review/active-learning/CI closure enhancements
- history/3D evaluation-prep reporting

The implementation keeps `Hybrid` as the primary DXF decision path and treats
`Graph2D` as a secondary signal for diagnostics, conflict detection, and review
prioritization.

## Implemented Changes

### 1. Output normalization in analyze

Added stable coarse-label outputs to the classification payload:

- `coarse_part_type`
- `coarse_fine_part_type`
- `coarse_hybrid_label`
- `coarse_graph2d_label`
- `coarse_filename_label`
- `coarse_titleblock_label`
- `coarse_history_label`
- `coarse_process_label`
- `coarse_part_family`
- `final_decision_source`
- `branch_conflicts`
- `has_branch_conflict`

Primary files:

- `src/api/v1/analyze.py`
- `src/core/classification/coarse_labels.py`
- `src/core/classification/__init__.py`

### 2. Knowledge productization output

Added lightweight structured knowledge signals driven by existing knowledge modules:

- `knowledge_checks`
- `violations`
- `standards_candidates`
- `knowledge_hints`

Covered signal families:

- metric thread designation detection
- ISO 2768 general tolerance designation detection
- IT grade detection
- GD&T frame interpretation
- datum-order warning generation
- dynamic knowledge hint surfacing

Primary file:

- `src/core/knowledge/analysis_summary.py`

### 3. Review / active-learning closure

Active-learning export and review-pack output now preserve richer triage context:

- `sample_type`
- `feedback_priority`
- coarse/fine label context
- rejection/knowledge-conflict context
- extended review-pack summary fields in CI

Primary files:

- `src/core/active_learning.py`
- `scripts/export_hybrid_rejection_review_pack.py`
- `.github/workflows/evaluation-report.yml`

### 4. History / 3D evaluation-prep

Manifest evaluation now emits report-friendly history and B-Rep prep signals:

- `history_label`
- `history_confidence`
- `history_status`
- `history_source`
- `history_shadow_only`
- `history_used_for_fusion`
- `history_input_resolved`
- `history_input_source`
- `brep_feature_hints`
- `brep_primary_surface_type`
- `brep_feature_hint_top_label`
- `brep_embedding_dim`

Primary files:

- `scripts/eval_hybrid_dxf_manifest.py`
- `src/ml/vision_3d.py`

## Verification

### Static validation

Commands:

```bash
python3 -m py_compile \
  src/api/v1/analyze.py \
  src/core/active_learning.py \
  src/core/classification/coarse_labels.py \
  src/core/knowledge/analysis_summary.py \
  src/ml/vision_3d.py \
  scripts/eval_hybrid_dxf_manifest.py \
  scripts/export_hybrid_rejection_review_pack.py

flake8 \
  src/api/v1/analyze.py \
  src/core/active_learning.py \
  src/core/classification/__init__.py \
  src/core/classification/coarse_labels.py \
  src/core/knowledge/analysis_summary.py \
  src/ml/vision_3d.py \
  scripts/eval_hybrid_dxf_manifest.py \
  scripts/export_hybrid_rejection_review_pack.py \
  tests/unit/test_classification_coarse_labels.py \
  tests/unit/test_knowledge_analysis_summary.py \
  tests/unit/test_active_learning_export_context.py \
  tests/unit/test_active_learning_loop.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  tests/unit/test_vision_3d_reporting.py \
  tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py \
  --max-line-length=100
```

Result:

- passed

### Automated tests

Command:

```bash
pytest -q \
  tests/unit/test_classification_coarse_labels.py \
  tests/unit/test_knowledge_analysis_summary.py \
  tests/unit/test_active_learning_export_context.py \
  tests/unit/test_active_learning_loop.py \
  tests/unit/test_export_hybrid_rejection_review_pack.py \
  tests/unit/test_export_hybrid_rejection_review_pack_context.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  tests/unit/test_vision_3d_reporting.py \
  tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py \
  tests/integration/test_analyze_dxf_hybrid_explanation.py \
  tests/integration/test_analyze_dxf_fusion.py \
  tests/integration/test_analyze_dxf_graph2d_prediction_contract.py \
  tests/test_active_learning_api.py \
  tests/test_l3_fusion_flow.py
```

Result:

- `50 passed`

Notes:

- third-party dependency warnings were present from `httpx`, `python_multipart`,
  `ezdxf`, and `torch_geometric`; no test failure was caused by them

### Real-data DXF validation sample

Command:

```bash
python3 scripts/eval_hybrid_dxf_manifest.py \
  --dxf-dir '/Users/huazhou/Downloads/训练图纸/训练图纸_dxf' \
  --manifest reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_CLEANED_20260123.csv \
  --max-files 10 \
  --output-dir reports/experiments/20260306/ai_parallel_validation_sample
```

Result summary:

- sample size: `10`
- `final_part_type` accuracy: `0.4`
- `graph2d_label` accuracy: `0.6`
- `filename_label` accuracy: `1.0`
- `titleblock_label` accuracy: `1.0`
- `hybrid_label` accuracy: `1.0`
- `fine_part_type` accuracy: `1.0`
- `graph2d_label.low_conf_rate`: `1.0`

Prep-signal summary:

- `history_prediction_count = 0`
- `history_input_resolved_count = 0`
- `brep_valid_3d_count = 0`
- `brep_feature_hints_count = 0`

Interpretation:

- the hybrid path remains the strongest DXF classification signal on this sample
- Graph2D remains a weak/diagnostic branch rather than a trustworthy final decision source
- the new eval/report path correctly preserves history/3D prep fields even when no
  history or B-Rep input is present

## Files Added or Updated

Core implementation:

- `src/api/v1/analyze.py`
- `src/core/active_learning.py`
- `src/core/classification/__init__.py`
- `src/core/classification/coarse_labels.py`
- `src/core/knowledge/analysis_summary.py`
- `src/ml/vision_3d.py`
- `scripts/eval_hybrid_dxf_manifest.py`
- `scripts/export_hybrid_rejection_review_pack.py`
- `.github/workflows/evaluation-report.yml`

Tests:

- `tests/unit/test_classification_coarse_labels.py`
- `tests/unit/test_knowledge_analysis_summary.py`
- `tests/unit/test_active_learning_export_context.py`
- `tests/unit/test_active_learning_loop.py`
- `tests/unit/test_eval_hybrid_dxf_manifest.py`
- `tests/unit/test_vision_3d_reporting.py`
- `tests/unit/test_export_hybrid_rejection_review_pack.py`
- `tests/unit/test_export_hybrid_rejection_review_pack_context.py`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- `tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py`

## Remaining Gaps

- no true `.h5` history dataset was provided in this run, so history real-data validation
  is still pending
- no true STEP/B-Rep production dataset was exercised in this run, so 3D prep coverage
  is currently limited to unit and report-path validation
- `Graph2D` accuracy remains materially weaker than `hybrid`, so it should continue to
  be treated as a secondary signal
