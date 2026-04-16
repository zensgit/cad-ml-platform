# Phase 3 Quality Pipeline Extraction Development Plan

## Goal
- Extract the inline `quality / DFM` block from `src/api/v1/analyze.py` into a shared helper without changing the current DFM-first / fallback behavior.

## Scope
- Move the `_run_quality()` business logic into `src/core/dfm/quality_pipeline.py`.
- Keep `analyze.py` responsible only for task scheduling, result writeback, and stage timing.
- Preserve the current output contract for all three paths:
  - DFM success
  - DFM exception fallback
  - non-3D quality fallback

## Planned Changes
- Add `run_quality_pipeline(...)` to orchestrate:
  - optional DFM path when `features_3d` is present
  - late-bound `classification.part_type` lookup via getter, so parallel classification timing semantics stay unchanged
  - DFM latency metric observation
  - fallback to `CADAnalyzer.check_quality(...)`
- Replace the inline `_run_quality()` body in `src/api/v1/analyze.py` with a single helper call.
- Add unit coverage for:
  - DFM happy path
  - missing `thin_walls_detected` geometry probe path
  - DFM exception fallback
  - non-DFM normalization path
  - `features_3d=None`
  - `classification_payload_getter -> None`
  - incomplete DFM payload fallback
- Add an integration lock proving the analyze route now delegates quality handling to the shared helper.

## Risk Controls
- Preserve the old late-read behavior for `results["classification"]` by passing a getter instead of a snapshot payload.
- Preserve the old best-effort DFM exception fallback shape, even if it differs from the normalized non-DFM path.
- Keep metric observation in the caller-visible path by passing `dfm_analysis_latency_seconds.observe` into the helper.
- Verify both existing integration coverage and parallel-task metric tests after extraction.
