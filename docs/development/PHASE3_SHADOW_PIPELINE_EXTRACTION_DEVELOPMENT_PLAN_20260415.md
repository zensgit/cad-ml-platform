# Phase 3 Shadow Pipeline Extraction Development Plan

## Goal
- Extract the remaining shadow evidence collection block from `src/api/v1/analyze.py` into a shared classification helper without changing final decision order.

## Scope
- Move the `ml_result`, `graph2d_result`, `graph2d_fusable`, `hybrid_result`, `part_classifier_prediction`, and `soft_override_suggestion` collection path into `src/core/classification/shadow_pipeline.py`.
- Keep `FusionAnalyzer` override, `Hybrid` override, finalization, and active-learning dispatch in `analyze.py`.
- Preserve existing helper imports from `src.api.v1.analyze` so current tests keep working.

## Planned Changes
- Add `build_shadow_classification_context(...)` to orchestrate:
  - ML overlay
  - Graph2D shadow prediction and enrich gates
  - Hybrid shadow prediction and history sidecar resolution
  - Part classifier shadow prediction
  - Graph2D soft override suggestion
- Re-export the new orchestration helper from `src/core/classification/__init__.py`.
- Replace the inline shadow block in `src/api/v1/analyze.py` with a single helper call.
- Add unit coverage for the extracted shadow pipeline.
- Add one integration case to lock the `Graph2D -> Fusion l4` precedence path.

## Risk Controls
- Keep existing `_enrich_graph2d_prediction`, `_build_graph2d_soft_override_suggestion`, and `_resolve_history_sequence_file_path` tests alive through `src.api.v1.analyze` aliases.
- Preserve payload field names and write order for:
  - `graph2d_prediction`
  - `history_sequence_input`
  - `hybrid_decision`
  - `hybrid_rejection`
  - `fine_part_type`
  - `part_classifier_prediction`
  - `soft_override_suggestion`
- Verify that `fusion_inputs["l4"]` still prefers Graph2D over ML when `GRAPH2D_FUSION_ENABLED=true`.
