# Phase 3 Analysis Preflight Extraction Development Plan

## Goal

Extract the request/file preflight block from `src/api/v1/analyze.py` into a
shared helper while preserving:

- `options` JSON parsing into `AnalysisOptions`
- single-content-hash cache key behavior
- optional PartClassifier shadow cache-key suffix behavior
- cache hit / miss metrics and sliding-window gauges
- cached-response contract returned by `/api/v1/analyze`
- downstream handoff of the same `content` bytes to `document_pipeline`

## Scope

### In

- add `src/core/analysis_preflight.py`
- move cache-key construction into a shared helper
- move cache hit / miss lookup and metrics updates
- move cached-response payload assembly for `/api/v1/analyze`
- keep route-level orchestration as a thin caller

### Out

- document validation / adaptation (`document_pipeline`)
- feature / classification / quality / process / vector flows
- route path or response schema changes

## Design

Create `run_analysis_request_preflight(...)` with:

- `file_name`
- raw `options` string
- already-read `content`
- `analysis_id`
- request timestamp
- injectable options model class and cache getter for unit tests

Return a context dict containing:

- `analysis_options`
- `analysis_cache_key`
- `cached`
- `cached_response`

`analyze.py` keeps:

- request ID / timestamp setup
- one `await file.read()` call
- calling the preflight helper
- early return on cache hit
- all downstream pipeline orchestration on cache miss

## Risk Controls

- preserve cache isolation for same-name different-content uploads
- preserve shadow-cache isolation across provider env toggles
- keep cached-response fields unchanged: `processing_time=0.1`,
  `cache_hit=true`, `cad_document=None`
- validate with existing route-level cache hash and cache metrics tests

## Validation Plan

1. `python3 -m py_compile src/core/analysis_preflight.py src/api/v1/analyze.py tests/unit/test_analysis_preflight.py`
2. `.venv311/bin/flake8 src/core/analysis_preflight.py src/api/v1/analyze.py tests/unit/test_analysis_preflight.py`
3. `.venv311/bin/python -m pytest -q tests/unit/test_analysis_preflight.py tests/unit/test_analysis_cache_hash.py tests/unit/test_analysis_cache_metrics.py tests/unit/test_document_pipeline.py tests/test_api_integration.py`
