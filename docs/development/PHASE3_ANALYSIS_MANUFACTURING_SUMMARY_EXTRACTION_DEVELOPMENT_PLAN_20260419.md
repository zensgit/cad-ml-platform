# Phase 3 Analyze Manufacturing Summary Extraction Development Plan

## Goal
- Extract the `manufacturing_decision` post-processing block from `src/api/v1/analyze.py`.
- Keep route behavior and analyze-level patch points unchanged.

## Scope
- Add `src/core/analysis_manufacturing_summary.py`
- Update `src/api/v1/analyze.py` to delegate summary attachment
- Add direct unit coverage for the shared helper

## Constraints
- Do not move the real summary strategy out of `src.core.process`
- Preserve `src.api.v1.analyze.build_manufacturing_decision_summary` as the patch point
- Preserve warning-only failure behavior

## Validation Plan
- `py_compile` on touched files
- `flake8` on touched files
- Focused `pytest` for helper unit tests and analyze integration coverage
