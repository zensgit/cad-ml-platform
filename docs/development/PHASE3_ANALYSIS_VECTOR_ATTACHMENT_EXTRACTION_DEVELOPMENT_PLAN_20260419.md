# Phase 3 Analyze Vector Attachment Extraction Development Plan

## Goal
- Extract the `analyze.py` vector post-processing wrapper into a shared helper.
- Keep route behavior and analyze-level vector pipeline patch points unchanged.

## Scope
- Add `src/core/analysis_vector_attachment.py`
- Update `src/api/v1/analyze.py` to delegate vector context attachment
- Add focused unit coverage for similarity write-back and stage timing

## Constraints
- Preserve `src.api.v1.analyze.run_vector_pipeline`
- Do not move the real vector registration/similarity logic
- Keep similarity stage timing semantics unchanged

## Validation Plan
- `py_compile` on touched files
- `flake8` on touched files
- Focused `pytest` for the new helper and existing analyze vector integration coverage
