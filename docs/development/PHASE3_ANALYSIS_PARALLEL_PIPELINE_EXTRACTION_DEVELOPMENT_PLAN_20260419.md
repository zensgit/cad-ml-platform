# Phase 3 Analyze Parallel Pipeline Extraction Development Plan

## Goal
- Extract the `analyze.py` parallel orchestration for classification, quality, and process into a shared helper.
- Keep route behavior and existing monkeypatch points stable.

## Scope
- Add `src/core/analysis_parallel_pipeline.py`
- Update `src/api/v1/analyze.py` to delegate parallel execution
- Add focused unit coverage for the new helper

## Constraints
- Do not move classification, quality, or process decision logic again
- Preserve `src.api.v1.analyze.run_classification_pipeline`
- Preserve `src.api.v1.analyze.run_quality_pipeline`
- Preserve `src.api.v1.analyze.run_process_pipeline`
- Preserve existing metrics semantics

## Validation Plan
- `py_compile` on touched files
- `flake8` on touched files
- Focused `pytest` for the new helper and affected analyze integration tests
