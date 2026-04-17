# Phase 3 Feature Pipeline Extraction Development Plan

## Goal

Extract the feature extraction orchestration block from
`src/api/v1/analyze.py` into a shared helper while preserving:

- request and response schemas
- 3D feature cache behavior
- 2D feature cache behavior
- `results["features"]` / `results["features_3d"]` payload shape
- downstream pipeline inputs for classification, quality, process, and vector registration

## Scope

### In

- add `src/core/feature_pipeline.py`
- move 3D feature extraction + cache orchestration
- move 2D feature extraction + cache orchestration
- move feature result payload assembly
- keep `analyze.py` as a thin caller that merges helper outputs

### Out

- input validation / adapter parsing
- downstream classification / quality / process / vector flows
- route path or schema changes

## Design

Create `run_feature_pipeline(...)` with:

- feature extraction toggle
- file format / file name / content / doc inputs
- stage timing context input
- optional dependency injection for feature extractor, caches, geometry engine, and 3D encoder

Return a context dict containing:

- `features`
- `features_3d`
- `results_patch`
- `features_stage_duration`
- `features_3d_stage_duration`

`analyze.py` keeps:

- parse/input lifecycle
- calling `run_feature_pipeline(...)`
- merging `results_patch`
- writing feature stage timings into `stage_times`
- observing existing stage metrics

## Risk Controls

- preserve existing swallow-on-error behavior for the 3D feature path
- preserve 2D feature cache hit/rehydrate behavior
- preserve cache lookup metrics and cache size updates
- validate against route-level integration tests that already lock the downstream behavior

## Validation Plan

1. `python3 -m py_compile src/core/feature_pipeline.py src/api/v1/analyze.py tests/unit/test_feature_pipeline.py`
2. `.venv311/bin/flake8 src/core/feature_pipeline.py src/api/v1/analyze.py tests/unit/test_feature_pipeline.py`
3. `.venv311/bin/python -m pytest -q tests/unit/test_feature_pipeline.py tests/unit/test_feature_cache.py tests/unit/test_feature_slots.py tests/integration/test_analyze_vector_pipeline.py tests/integration/test_analyze_quality_pipeline.py tests/integration/test_analyze_process_pipeline.py tests/integration/test_analyze_manufacturing_summary.py tests/test_api_integration.py`
