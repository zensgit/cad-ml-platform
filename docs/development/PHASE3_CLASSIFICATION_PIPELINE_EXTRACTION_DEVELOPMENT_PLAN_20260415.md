# Phase 3 Classification Pipeline Extraction Development Plan

## Goal
- Extract the remaining end-to-end classify orchestration from `src/api/v1/analyze.py` into a shared async pipeline helper.

## Scope
- Move the `baseline -> shadow -> fusion -> hybrid -> finalization -> active-learning dispatch` chain into `src/core/classification/classification_pipeline.py`.
- Keep request-level timing, result assignment, and parallel task wiring in `analyze.py`.
- Reuse the already-extracted helper modules as the only underlying policy and orchestration building blocks.

## Planned Changes
- Add `run_classification_pipeline(...)` to orchestrate:
  - baseline context
  - shadow context
  - fusion context with degraded logging on failure
  - hybrid override context
  - finalization with review thresholds
  - active-learning flag dispatch with degraded logging on failure
- Re-export the new helper from `src/core/classification/__init__.py`.
- Replace the inline classify chain in `src/api/v1/analyze.py` with a single helper call.
- Add async unit coverage for the orchestration helper.

## Risk Controls
- Preserve the current call order:
  - baseline
  - shadow
  - fusion
  - hybrid
  - finalization
  - active learning flag
- Preserve FusionAnalyzer failure as best-effort logging, not request failure.
- Preserve active-learning flag failure as warning-only, not request failure.
- Keep finalization thresholds sourced from the same env vars as before.
