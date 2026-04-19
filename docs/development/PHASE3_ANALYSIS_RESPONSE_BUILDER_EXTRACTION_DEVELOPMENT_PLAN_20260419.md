# Phase 3 Analyze Response Builder Extraction Development Plan

## Goal
- Extract the final `finalize_analysis_success` route wrapper from `src/api/v1/analyze.py`.
- Keep route behavior and analyze-level finalize patch points unchanged.

## Scope
- Add `src/core/analysis_response_builder.py`
- Update `src/api/v1/analyze.py` to delegate final response construction
- Add direct unit coverage for the shared helper

## Constraints
- Preserve `src.api.v1.analyze.finalize_analysis_success`
- Keep `AnalysisResult(**payload)` wrapping behavior unchanged
- Do not move the real finalize/persist logic out of `analysis_result_envelope`

## Validation Plan
- `py_compile` on touched files
- `flake8` on touched files
- Focused `pytest` for the new helper and existing analyze result envelope integration coverage
