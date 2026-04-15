# Phase 3 Hybrid Override Pipeline Extraction Development Plan

## Goal
- Extract the remaining Hybrid override orchestration block from `src/api/v1/analyze.py` into a shared classification helper without changing override semantics.

## Scope
- Move `HYBRID_CLASSIFIER_OVERRIDE`, `HYBRID_CLASSIFIER_AUTO_OVERRIDE`, `HYBRID_OVERRIDE_MIN_CONF`, and `HYBRID_OVERRIDE_BASE_MAX_CONF` env handling plus `apply_hybrid_override(...)` invocation into `src/core/classification/hybrid_override_pipeline.py`.
- Keep finalization, review gating, and active-learning dispatch in `analyze.py`.
- Preserve existing `apply_hybrid_override(...)` policy logic and mode names.

## Planned Changes
- Add `build_hybrid_override_context(...)` to orchestrate:
  - env-driven Hybrid override enablement
  - threshold parsing
  - `apply_hybrid_override(...)` delegation
- Re-export the new helper from `src/core/classification/__init__.py`.
- Replace the inline Hybrid override block in `src/api/v1/analyze.py` with a single helper call.
- Add unit coverage for the new helper.
- Add one integration case to lock the env-forced override path.

## Risk Controls
- Preserve `hybrid_override_applied.mode` values:
  - `env`
  - `auto`
  - `auto_low_conf`
  - `auto_drawing_type`
- Preserve `hybrid_override_skipped` behavior for low-confidence env overrides.
- Keep `apply_hybrid_override(...)` as the only place that mutates final classification fields for Hybrid override.
