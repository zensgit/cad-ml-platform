# /analyze DWG Fail-Closed Validation - 2026-07-17

## Scope

This slice closes GitHub issue #524 as a safety guardrail from VemCAD #995. It
does not start the VemCAD Day 0 / P0 product execution track.

## Problem

Raw `.dwg` input could reach `/api/v1/analyze`, be routed through the DXF
adapter path, and degrade into an empty successful analysis when parsing failed.
That is fail-open behavior: downstream consumers can receive a successful
analysis envelope even though the drawing was never parsed.

## Change

- `/analyze` now rejects raw DWG input before adapter dispatch with HTTP 415 and
  error code `UNSUPPORTED_INPUT_DWG`.
- DXF parser-stub output for payloads that do not contain a DXF `SECTION` marker
  now fails closed with HTTP 422 `PARSE_FAILED` instead of returning an empty
  success. This keeps legacy lightweight DXF fixtures usable for non-parser unit
  tests while still rejecting arbitrary non-DXF bytes.
- The existing entity-limit test now uses a valid one-entity DXF fixture, so it
  still tests entity validation rather than parser-stub behavior.

## Verification

Commands run in a temporary Python 3.11 virtualenv:

```bash
.venv/bin/python -m pytest tests/unit/test_error_codes_analysis.py -q
.venv/bin/python -m pytest \
  tests/unit/test_signature_validation_fail.py \
  tests/unit/test_strict_format_validation.py \
  tests/unit/test_document_pipeline.py -q
.venv/bin/python -m pytest \
  tests/unit/test_error_codes_analysis.py \
  tests/unit/test_signature_validation_fail.py \
  tests/unit/test_strict_format_validation.py \
  tests/unit/test_document_pipeline.py \
  tests/unit/test_analysis_cache_hash.py \
  tests/unit/test_analysis_cache_metrics.py \
  tests/unit/test_feature_cache.py \
  tests/unit/test_feature_slots.py \
  tests/unit/test_format_matrix_exempt.py \
  tests/unit/test_parallel_execution_metric.py \
  tests/unit/test_similarity_complexity_filter.py \
  tests/unit/test_similarity_endpoint.py \
  tests/unit/test_similarity_error_codes.py \
  tests/unit/test_similarity_filters.py \
  tests/unit/test_similarity_topk.py \
  tests/unit/test_similarity_topk_pagination.py \
  tests/unit/test_vector_management.py \
  tests/unit/test_vector_update_dimension_conflict.py -q
```

Results:

- `tests/unit/test_error_codes_analysis.py`: 5 passed.
- Signature, strict-format, and document-pipeline slice: 10 passed.
- Combined related slice: 15 passed.
- Full targeted fail-closed + affected success-path slice: 39 passed.

The local Python 3.11 environment required the repo's lightweight test
dependencies plus `pyyaml` and `python-multipart` to import `src.main`.
