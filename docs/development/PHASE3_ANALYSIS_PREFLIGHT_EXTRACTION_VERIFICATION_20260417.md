# Phase 3 Analysis Preflight Extraction Verification

## Scope Verified

Verified extraction of the request/file preflight block from `analyze.py` into
`src/core/analysis_preflight.py`, including:

- `AnalysisOptions` parsing from raw JSON
- content-hash-based analysis cache key construction
- optional PartClassifier shadow cache-key suffix behavior
- analysis cache hit / miss metrics updates
- cached-response payload contract for early return
- reuse of the same `content` bytes on cache miss

## Files Verified

- `src/core/analysis_preflight.py`
- `src/api/v1/analyze.py`
- `tests/unit/test_analysis_preflight.py`

## Commands Run

### Static validation

```bash
python3 -m py_compile \
  src/core/analysis_preflight.py \
  src/api/v1/analyze.py \
  tests/unit/test_analysis_preflight.py
```

Result: pass

```bash
.venv311/bin/flake8 \
  src/core/analysis_preflight.py \
  src/api/v1/analyze.py \
  tests/unit/test_analysis_preflight.py
```

Result: pass

### Regression validation

```bash
.venv311/bin/python -m pytest -q \
  tests/unit/test_analysis_preflight.py \
  tests/unit/test_analysis_cache_hash.py \
  tests/unit/test_analysis_cache_metrics.py \
  tests/unit/test_document_pipeline.py \
  tests/test_api_integration.py
```

Result: `13 passed, 7 warnings`

## Outcome

The analyze request preflight is now centralized in a shared helper while
`analyze.py` keeps only a single `file.read()`, one helper call, and the early
cache-hit return. The validated regression set confirms no drift in cache-key
content hashing, cache hit / miss metrics exposure, document-pipeline handoff,
or the selected API integration path.
