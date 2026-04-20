# Phase 3 Analyze Batch Router Extraction Development Plan

## Goal
- Split the live `/api/v1/analyze/batch` and `/api/v1/analyze/batch-classify` routes out of `src/api/v1/analyze.py`.
- Keep route behavior unchanged while further shrinking the main analyze route module.

## Scope
- Add `src/api/v1/analyze_batch_router.py`
- Update `src/api/v1/analyze.py` to include the split batch router
- Update direct integration patch tests to target the new router module
- Add route ownership smoke coverage

## Constraints
- Preserve route paths and request/response contracts
- Preserve `analyze_cad_file` delegation for `/batch`
- Keep changes limited to the two batch routes only

## Validation Plan
- `py_compile` on touched files
- `flake8` on touched files
- Focused `pytest` for batch route ownership, batch integration patch tests, and batch-classify endpoint coverage
