# Phase 3 Analyze Drift Attachment Extraction Development Plan

## Goal
- Extract the `analyze.py` drift post-processing wrapper into a shared helper.
- Keep route behavior, swallow-on-failure semantics, and analyze-level patch points unchanged.

## Scope
- Add `src/core/analysis_drift_attachment.py`
- Update `src/api/v1/analyze.py` to delegate drift attachment
- Add focused unit coverage for the helper

## Constraints
- Preserve `src.api.v1.analyze.run_analysis_drift_pipeline`
- Preserve `src.api.v1.analyze._DRIFT_STATE`
- Keep cache lookup inside the helper and keep failures non-fatal

## Validation Plan
- `py_compile` on touched files
- `flake8` on touched files
- Focused `pytest` for the new helper and existing drift coverage
