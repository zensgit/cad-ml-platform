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
- DXF parser-stub output with zero entities and no layers now fails closed with
  HTTP 422 `PARSE_FAILED` instead of returning an empty success.
- The existing entity-limit test now uses a valid one-entity DXF fixture, so it
  still tests entity validation rather than parser-stub behavior.

## Verification

Commands run in a temporary Python 3.11 virtualenv:

```bash
.venv/bin/python -m pytest tests/unit/test_error_codes_analysis.py -q
.venv/bin/python -m pytest tests/unit/test_signature_validation_fail.py tests/unit/test_strict_format_validation.py tests/unit/test_document_pipeline.py -q
```

Results:

- `tests/unit/test_error_codes_analysis.py`: 5 passed.
- `tests/unit/test_signature_validation_fail.py`, `tests/unit/test_strict_format_validation.py`, `tests/unit/test_document_pipeline.py`: 10 passed.
- Combined related slice: 15 passed.

The local Python 3.11 environment required the repo's lightweight test
dependencies plus `pyyaml` and `python-multipart` to import `src.main`.
