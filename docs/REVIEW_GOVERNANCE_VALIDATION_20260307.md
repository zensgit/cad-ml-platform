# Review Governance Validation 2026-03-07

## Goal
Stabilize review governance outputs across API, evaluation, review-pack, and active-learning so the system exposes a consistent decision-governance layer instead of scattered ad hoc logic.

## Scope
Implemented:
- Shared review governance helpers in `src/core/classification/review_governance.py`
- Unified classification outputs in `src/api/v1/analyze.py`
- Active-learning priority/sample-type derivation updates
- DXF manifest evaluation export + summary updates
- Review-pack export + summary updates
- Evaluation workflow summary / PR-comment extensions

## Main Outputs Added
Classification payload now exposes:
- `needs_review`
- `confidence_band`
- `review_priority`
- `review_priority_score`
- `review_reasons`

Review-pack / eval artifacts now surface:
- `review_priority`
- `review_confidence_band`
- aggregated top priorities / confidence bands

## Files
Core:
- `src/core/classification/review_governance.py`
- `src/core/classification/__init__.py`
- `src/api/v1/analyze.py`
- `src/core/active_learning.py`

Reporting:
- `scripts/batch_analyze_dxf_local.py`
- `scripts/eval_hybrid_dxf_manifest.py`
- `scripts/export_hybrid_rejection_review_pack.py`
- `.github/workflows/evaluation-report.yml`

Tests:
- `tests/unit/test_review_governance.py`
- `tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py`
- `tests/unit/test_eval_hybrid_dxf_manifest.py`
- `tests/unit/test_export_hybrid_rejection_review_pack.py`
- `tests/unit/test_export_hybrid_rejection_review_pack_context.py`
- `tests/unit/test_active_learning_export_context.py`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation Commands
```bash
python3 -m py_compile \
  src/core/classification/review_governance.py \
  src/api/v1/analyze.py \
  src/core/active_learning.py \
  scripts/eval_hybrid_dxf_manifest.py \
  scripts/export_hybrid_rejection_review_pack.py \
  scripts/batch_analyze_dxf_local.py \
  tests/unit/test_review_governance.py \
  tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  tests/unit/test_export_hybrid_rejection_review_pack.py \
  tests/unit/test_export_hybrid_rejection_review_pack_context.py \
  tests/unit/test_active_learning_export_context.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 \
  src/core/classification/review_governance.py \
  src/api/v1/analyze.py \
  src/core/active_learning.py \
  scripts/eval_hybrid_dxf_manifest.py \
  scripts/export_hybrid_rejection_review_pack.py \
  scripts/batch_analyze_dxf_local.py \
  tests/unit/test_review_governance.py \
  tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  tests/unit/test_export_hybrid_rejection_review_pack.py \
  tests/unit/test_export_hybrid_rejection_review_pack_context.py \
  tests/unit/test_active_learning_export_context.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_review_governance.py \
  tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  tests/unit/test_export_hybrid_rejection_review_pack.py \
  tests/unit/test_export_hybrid_rejection_review_pack_context.py \
  tests/unit/test_active_learning_export_context.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Validation Result
- `py_compile`: passed
- `flake8`: passed
- `pytest`: `19 passed`

## Notes
- Warnings observed during pytest came from third-party dependencies (`python_multipart`, `httpx`, `ezdxf`, `torch_geometric`) and were not introduced by this change.
- No model weights or decision thresholds were retrained here; this is a governance/output stabilization change.

## Outcome
The system now emits a consistent review-governance layer that can be consumed by:
- API clients
- active-learning export
- DXF manifest evaluation
- review-pack prioritization
- GitHub Actions job summaries / PR comments

This directly improves explainability and makes the AI stack more suitable for industrial review workflows.
