# Phase 3 Fusion Pipeline Extraction Development Plan

## Goal
- Extract the remaining `FusionAnalyzer` orchestration block from `src/api/v1/analyze.py` into a shared classification helper without changing decision order.

## Scope
- Move `FusionAnalyzer` enablement, L4 source selection, analyzer invocation, `fusion_decision`/`fusion_inputs` writeback, and fusion override application into `src/core/classification/fusion_pipeline.py`.
- Keep `Hybrid` override, finalization, and active-learning dispatch in `analyze.py`.
- Preserve payload field names and current override semantics.

## Planned Changes
- Add `build_fusion_classification_context(...)` to orchestrate:
  - env-driven FusionAnalyzer enablement
  - `graph2d_fusable -> ml_result -> None` L4 selection
  - `FusionAnalyzer.analyze(...)` invocation
  - `fusion_decision` and `fusion_inputs` payload writeback
  - `apply_fusion_override(...)` integration
- Re-export the new helper from `src/core/classification/__init__.py`.
- Replace the inline fusion block in `src/api/v1/analyze.py` with a single helper call inside the existing `try/except` boundary.
- Add unit coverage for the extracted fusion pipeline.
- Add one integration case to lock the end-to-end fusion override decision contract.

## Risk Controls
- Keep the existing caller-side `try/except` in `analyze.py` so FusionAnalyzer failures still degrade to best-effort logging.
- Preserve `fusion_inputs["l4"]` selection order:
  - `graph2d_fusable` when `GRAPH2D_FUSION_ENABLED=true`
  - otherwise `ml_result.predicted_type`
  - otherwise `None`
- Preserve `apply_fusion_override(...)` guardrails for:
  - below-threshold decisions
  - `RULE_DEFAULT` rule-only fallback decisions
- Verify that `FusionAnalyzer-v<schema>` still becomes the final `rule_version` when override applies.
