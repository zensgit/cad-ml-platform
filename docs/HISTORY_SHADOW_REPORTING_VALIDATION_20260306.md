# History Shadow Reporting Validation

Date: 2026-03-06

## Scope

This update makes `history_sequence` shadow-only evidence visible across:

- analyze-time active-learning capture
- exported active-learning training data
- local batch DXF CSV output
- hybrid rejection review-pack summaries

## New Reporting Fields

### Active Learning `score_breakdown`

Added:

- `history_prediction`
- `shadow_predictions`

These fields complement existing:

- `decision_path`
- `source_contributions`
- `fusion_metadata`
- `hybrid_explanation`

### Batch DXF CSV

Added:

- `history_label`
- `history_confidence`
- `history_status`
- `history_shadow_only`
- `history_used_for_fusion`
- `hybrid_shadow_predictions`

### Review Pack CSV / Summary

Added:

- `review_shadow_sources`
- `review_history_shadow_only`
- `review_history_shadow_label`
- `review_history_shadow_confidence`

Added to summary JSON:

- `top_shadow_sources`

## Validation

Executed:

```bash
python3 -m py_compile \
  src/api/v1/analyze.py \
  src/core/active_learning.py \
  scripts/batch_analyze_dxf_local.py \
  scripts/export_hybrid_rejection_review_pack.py \
  tests/integration/test_analyze_dxf_active_learning_context.py \
  tests/unit/test_active_learning_export_context.py \
  tests/unit/test_export_hybrid_rejection_review_pack_context.py

flake8 \
  src/api/v1/analyze.py \
  src/core/active_learning.py \
  scripts/batch_analyze_dxf_local.py \
  scripts/export_hybrid_rejection_review_pack.py \
  tests/integration/test_analyze_dxf_active_learning_context.py \
  tests/unit/test_active_learning_export_context.py \
  tests/unit/test_export_hybrid_rejection_review_pack_context.py \
  --max-line-length=100

pytest -q \
  tests/integration/test_analyze_dxf_active_learning_context.py \
  tests/unit/test_active_learning_export_context.py \
  tests/unit/test_export_hybrid_rejection_review_pack_context.py \
  tests/unit/test_hybrid_classifier_history_branch.py \
  tests/unit/test_hybrid_config_loader.py
```

Result:

- `14 passed`

## Notes

- Shadow evidence is observable but still non-blocking for final label selection.
- This keeps rollout risk low while making review data richer for future fusion decisions.
