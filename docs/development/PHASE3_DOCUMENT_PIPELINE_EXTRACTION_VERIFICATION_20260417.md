# Phase 3 Document Pipeline Extraction Verification

## Scope Verified

Verified extraction of the early document ingestion block from `analyze.py` into
`src/core/document_pipeline.py`, including:

- input validation and file format resolution
- adapter `parse` / legacy `convert` orchestration
- parse timeout handling
- signature / strict deep-format / matrix validation
- material / project metadata attachment
- parse-stage duration return and entity-count guard

## Files Verified

- `src/core/document_pipeline.py`
- `src/api/v1/analyze.py`
- `tests/unit/test_document_pipeline.py`

## Commands Run

### Static validation

```bash
python3 -m py_compile \
  src/core/document_pipeline.py \
  src/api/v1/analyze.py \
  tests/unit/test_document_pipeline.py
```

Result: pass

```bash
.venv311/bin/flake8 \
  src/core/document_pipeline.py \
  src/api/v1/analyze.py \
  tests/unit/test_document_pipeline.py
```

Result: pass

### Regression validation

```bash
.venv311/bin/python -m pytest -q \
  tests/unit/test_document_pipeline.py \
  tests/unit/test_parse_timeout.py \
  tests/unit/test_step_parse_failure.py \
  tests/unit/test_signature_validation.py \
  tests/unit/test_signature_validation_fail.py \
  tests/unit/test_strict_format_validation.py \
  tests/unit/test_format_matrix_exempt.py \
  tests/integration/test_analyze_quality_pipeline.py \
  tests/integration/test_analyze_process_pipeline.py \
  tests/integration/test_analyze_manufacturing_summary.py \
  tests/integration/test_analyze_vector_pipeline.py \
  tests/test_api_integration.py
```

Result: `21 passed, 7 warnings`

## Outcome

The document ingestion stage is now centralized in a shared helper while
`analyze.py` keeps only the route-level call site and downstream orchestration.
The validated regression set confirms no drift in timeout handling, invalid STEP
rejection, strict format validation, matrix-exempt behavior, or downstream
analyze flows covered by the selected route tests.
