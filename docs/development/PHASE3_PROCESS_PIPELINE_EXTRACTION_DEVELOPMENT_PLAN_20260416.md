# Phase 3 Process Pipeline Extraction Development Plan

## Goal
- Extract the inline `process recommendation + cost estimation` block from `src/api/v1/analyze.py` into a shared helper without changing the current AI-first / fallback behavior.

## Scope
- Move `_run_process()` business logic into `src/core/process/process_pipeline.py`.
- Keep `analyze.py` responsible for task scheduling, result writeback, and stage timing only.
- Preserve current behavior for:
  - AI process recommendation with 3D features
  - rule fallback when AI process fails
  - non-3D rule recommendation path
  - optional chained cost estimation

## Planned Changes
- Add `run_process_pipeline(...)` to orchestrate:
  - AI recommender path when `features_3d` is present
  - late-bound `classification.part_type` lookup via getter, so parallel timing semantics remain unchanged
  - rule fallback to `CADAnalyzer.recommend_process(...)`
  - optional cost estimation chained from the process result
  - rule-version metric observation on the non-3D rule path
- Re-export the helper from `src/core/process/__init__.py`.
- Replace the inline `_run_process()` body in `src/api/v1/analyze.py` with a single helper call.
- Add unit coverage for:
  - AI happy path with cost estimation
  - static `classification_payload`
  - getter returning `None`
  - AI failure fallback
  - non-3D rule path with `rule_version`
  - cost estimation failure
- Add an integration lock proving the analyze route now delegates process handling to the shared helper.

## Risk Controls
- Preserve the old late-read behavior for `results["classification"]` by passing a getter instead of a snapshot payload.
- Preserve the old raw `process` payload writeback shape on all fallback paths.
- Preserve the old cost-estimation chaining rule:
  - only run cost estimation when `estimate_cost=true` and `features_3d` is present
- Keep caller-visible metrics intact by passing:
  - `process_rule_version_total.labels(...).inc()`
  - `cost_estimation_latency_seconds.observe(...)`
  - while leaving `process_recommend_latency_seconds.observe(...)` in the caller
