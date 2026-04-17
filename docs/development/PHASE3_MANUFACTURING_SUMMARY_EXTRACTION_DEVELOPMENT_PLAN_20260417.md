# Phase 3 Manufacturing Summary Extraction Development Plan

## Goal
- Extract the inline `manufacturing_decision` summary block from `src/api/v1/analyze.py` into a shared helper without changing the current output contract.

## Scope
- Move the summary-building logic into `src/core/process/manufacturing_summary.py`.
- Keep `analyze.py` responsible only for collecting `quality`, `process`, and `cost_estimation` payloads and writing back the final summary.
- Preserve current behavior for:
  - emitting no `manufacturing_decision` when all three payloads are empty
  - deriving `process` from `primary_recommendation` first
  - falling back to legacy `{process, method}` shape when needed
  - computing `cost_range` as `total_unit_cost * 0.9 / 1.1`, rounded to 2 decimals

## Planned Changes
- Add `build_manufacturing_decision_summary(...)` to:
  - normalize optional quality / process / cost payloads
  - build the legacy-compatible `process` summary payload
  - compute `cost_range`
  - return `None` when no summary should be emitted
- Re-export the helper from `src/core/process/__init__.py`.
- Replace the inline summary block in `src/api/v1/analyze.py` with a single helper call.
- Add unit coverage for:
  - empty-input no-op behavior
  - full L4 quality + process + cost summary
  - legacy process fallback
  - empty `primary_recommendation` with legacy process fields present
- Add integration coverage for:
  - analyze route delegation to the shared summary helper
  - real helper wiring from `quality/process/cost_estimation` into `manufacturing_decision`

## Risk Controls
- Preserve the original output keys exactly:
  - `feasibility`
  - `risks`
  - `process`
  - `cost_estimate`
  - `cost_range`
  - `currency`
- Preserve the original defaulting behavior where missing `cost_estimation` still becomes `{}` inside the emitted summary when quality or process data exists.
- Keep exception handling in `analyze.py` so a summary failure still degrades to a warning instead of aborting the analysis response.
