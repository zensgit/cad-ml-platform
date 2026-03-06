# Hybrid AI Enhancements Validation (2026-03-06)

## Scope

This iteration strengthened the existing HybridClassifier pipeline in three areas:

1. Conditional branch auto-enable for `titleblock` and `history_sequence`.
2. Dynamic multi-source fusion with reusable fusion engines.
3. Explainable hybrid output surfaced through the analyze API.

## Implementation

### 1. Conditional auto-enable

Updated:
- `src/ml/hybrid_config.py`
- `config/hybrid_classifier.yaml`
- `src/ml/hybrid_classifier.py`

Added configuration:
- `auto_enable.titleblock_on_text`
- `auto_enable.history_on_path`

Behavior:
- `titleblock` branch now runs even when `TITLEBLOCK_ENABLED=false` if DXF text-like entities are present and `TITLEBLOCK_AUTO_ENABLE=true`.
- `history_sequence` branch now runs even when `HISTORY_SEQUENCE_ENABLED=false` if a resolved `.h5` path is provided and `HISTORY_SEQUENCE_AUTO_ENABLE=true`.

Decision-path markers added:
- `titleblock_auto_enabled`
- `history_auto_enabled`

### 2. Advanced fusion

Updated:
- `src/ml/hybrid_classifier.py`

Added configuration:
- `decision.advanced_fusion_enabled`
- `decision.fusion_strategy`
- `decision.auto_select_fusion`

Behavior:
- Multi-source predictions now use `src/ml/hybrid/fusion.py` when enabled.
- Existing direct-adopt rules remain intact for high-confidence filename / titleblock / history decisions.
- When advanced fusion runs, the classifier now emits:
  - `source_contributions`
  - `fusion_metadata`
  - fusion decision path markers such as `fusion_engine_weighted_average`

Fallback behavior:
- If advanced fusion is unavailable or fails, the previous manual weighted fusion path is preserved.

### 3. Explainable API output

Updated:
- `src/ml/hybrid/explainer.py`
- `src/api/v1/analyze.py`

Added configuration:
- `decision.explanation_enabled`

Behavior:
- `ClassificationResult` now carries:
  - `source_contributions`
  - `fusion_metadata`
  - `explanation`
- `/api/v1/analyze/` now surfaces:
  - `history_prediction`
  - `process_prediction`
  - `decision_path`
  - `source_contributions`
  - `fusion_metadata`
  - `hybrid_explanation`

Explainability updates:
- Added history-sequence feature analysis and contribution reporting.
- Added explanation coverage for new decision-path markers:
  - auto-enable markers
  - advanced fusion engine markers
  - reject-to-fallback marker

## Validation

### Static checks

```bash
python3 -m py_compile \
  src/ml/hybrid_config.py \
  src/ml/hybrid_classifier.py \
  src/ml/hybrid/explainer.py \
  src/api/v1/analyze.py
```

Result: pass.

```bash
flake8 \
  src/ml/hybrid_config.py \
  src/ml/hybrid_classifier.py \
  src/ml/hybrid/explainer.py \
  src/api/v1/analyze.py \
  tests/unit/test_hybrid_config_decision_auto_enable.py \
  tests/unit/test_hybrid_classifier_auto_explain.py \
  tests/integration/test_analyze_dxf_hybrid_explanation.py \
  --max-line-length=100
```

Result: pass.

### Pytest

```bash
pytest -q \
  tests/unit/test_hybrid_config_decision_auto_enable.py \
  tests/unit/test_hybrid_classifier_auto_explain.py \
  tests/unit/test_hybrid_classifier_history_branch.py \
  tests/unit/test_hybrid_classifier_rejection_policy.py \
  tests/unit/test_hybrid_config_loader.py \
  tests/integration/test_analyze_dxf_hybrid_explanation.py \
  tests/integration/test_analyze_dxf_fusion.py
```

Result: `23 passed`.

Observed warnings:
- `python_multipart` pending deprecation warning from third-party dependency.
- `httpx` TestClient deprecation warning (`app` shortcut).
- `ezdxf` / `pyparsing` deprecation warnings in dependency code.

No failures were produced in changed project code paths.

## Files Added

- `tests/unit/test_hybrid_config_decision_auto_enable.py`
- `tests/unit/test_hybrid_classifier_auto_explain.py`
- `tests/integration/test_analyze_dxf_hybrid_explanation.py`
- `docs/HYBRID_AI_ENHANCEMENTS_VALIDATION_20260306.md`

## Files Updated

- `config/hybrid_classifier.yaml`
- `src/ml/hybrid_config.py`
- `src/ml/hybrid_classifier.py`
- `src/ml/hybrid/explainer.py`
- `src/api/v1/analyze.py`

## Continuation: Review Pack / Active Learning Context

This continuation extended the explanation output into downstream human-review
and retraining flows.

### Updated

- `src/api/v1/analyze.py`
- `src/core/active_learning.py`
- `scripts/batch_analyze_dxf_local.py`
- `scripts/export_hybrid_rejection_review_pack.py`

### New behavior

- Active learning samples now persist richer hybrid context inside
  `score_breakdown`:
  - `decision_path`
  - `source_contributions`
  - `fusion_metadata`
  - `hybrid_explanation`
- Active-learning training exports now preserve:
  - `score_breakdown`
  - `uncertainty_reason`
- Local batch DXF analysis now flattens hybrid explanation fields into CSV:
  - `hybrid_source_contributions`
  - `hybrid_fusion_strategy`
  - `hybrid_fusion_agreement_score`
  - `hybrid_explanation_summary`
  - `hybrid_explanation`
- Review-pack export now derives analyst-facing context columns:
  - `review_primary_sources`
  - `review_explanation_summary`
  - `review_decision_path`
  - `review_fusion_strategy`

### Additional validation

```bash
python3 -m py_compile \
  src/api/v1/analyze.py \
  src/core/active_learning.py \
  scripts/batch_analyze_dxf_local.py \
  scripts/export_hybrid_rejection_review_pack.py
```

Result: pass.

```bash
flake8 \
  src/api/v1/analyze.py \
  src/core/active_learning.py \
  scripts/batch_analyze_dxf_local.py \
  scripts/export_hybrid_rejection_review_pack.py \
  tests/unit/test_active_learning_export_context.py \
  tests/unit/test_export_hybrid_rejection_review_pack_context.py \
  tests/integration/test_analyze_dxf_active_learning_context.py \
  --max-line-length=100
```

Result: pass.

```bash
pytest -q \
  tests/unit/test_active_learning_loop.py \
  tests/unit/test_active_learning_export_context.py \
  tests/unit/test_export_hybrid_rejection_review_pack.py \
  tests/unit/test_export_hybrid_rejection_review_pack_context.py \
  tests/integration/test_analyze_dxf_active_learning_context.py \
  tests/integration/test_analyze_dxf_hybrid_explanation.py \
  tests/integration/test_analyze_dxf_fusion.py
```

Result: `22 passed`.

## Continuation: CI Review-Pack Context Surfacing

This continuation pushed the explanation context one step further into GitHub
workflow outputs so reviewers can see why a sample is in the review pack without
opening the raw CSV first.

### Updated

- `.github/workflows/evaluation-report.yml`
- `scripts/export_hybrid_rejection_review_pack.py`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- `tests/unit/test_export_hybrid_rejection_review_pack.py`
- `tests/unit/test_export_hybrid_rejection_review_pack_context.py`

### New summary JSON fields

`export_hybrid_rejection_review_pack.py` now emits aggregated context fields:

- `top_review_reasons`
- `top_primary_sources`
- `sample_explanations`
- `sample_candidates`

### New workflow outputs and UI surfacing

The `Build hybrid rejection review pack (optional)` workflow step now exports:

- `top_review_reasons`
- `top_primary_sources`
- `sample_explanations`

These are now shown in:

- GitHub job summary
- PR comment block under `Graph2D Review Insights`

### Additional validation

```bash
python3 -m py_compile \
  scripts/export_hybrid_rejection_review_pack.py \
  src/core/active_learning.py \
  scripts/batch_analyze_dxf_local.py
```

Result: pass.

```bash
flake8 \
  scripts/export_hybrid_rejection_review_pack.py \
  src/core/active_learning.py \
  scripts/batch_analyze_dxf_local.py \
  tests/unit/test_export_hybrid_rejection_review_pack.py \
  tests/unit/test_export_hybrid_rejection_review_pack_context.py \
  tests/unit/test_active_learning_export_context.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/integration/test_analyze_dxf_active_learning_context.py \
  --max-line-length=100
```

Result: pass.

```bash
pytest -q \
  tests/unit/test_export_hybrid_rejection_review_pack.py \
  tests/unit/test_export_hybrid_rejection_review_pack_context.py \
  tests/unit/test_active_learning_loop.py \
  tests/unit/test_active_learning_export_context.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/integration/test_analyze_dxf_active_learning_context.py
```

Result: `16 passed`.
