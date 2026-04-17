# Phase 3 Document Pipeline Extraction Development Plan

## Goal

Extract the early document ingestion block from `src/api/v1/analyze.py` into a
shared helper while preserving:

- MIME / size / empty-file validation behavior
- file format allowlist behavior
- adapter `parse` / legacy `convert` fallback behavior
- parse timeout handling
- signature / strict deep-format / matrix validation behavior
- document metadata attachment and entity-count guard behavior
- existing downstream `doc`, `unified_data`, and `file_format` contract

## Scope

### In

- add `src/core/document_pipeline.py`
- move input validation and file format resolution
- move adapter orchestration and parse timeout handling
- move signature / strict validation / matrix validation
- move material / project metadata attachment
- move parse-stage duration / budget metrics and entity-count guard
- keep `analyze.py` as a thin caller that consumes a document context

### Out

- feature extraction pipeline
- classification / quality / process / vector flows
- route path or response schema changes

## Design

Create `run_document_pipeline(...)` with:

- file name / content / started time inputs
- optional material / project context
- optional adapter factory override for unit tests

Return a context dict containing:

- `file_format`
- `doc`
- `unified_data`
- `parse_stage_duration`

`analyze.py` keeps:

- cache lookup lifecycle
- calling `run_document_pipeline(...)`
- writing `stage_times["parse"]`
- all downstream pipeline orchestration

## Risk Controls

- preserve graceful fallback to empty `CadDocument` on non-timeout parse errors
- keep timeout behavior as `504`
- keep strict validation behind `FORMAT_STRICT_MODE`
- keep legacy `convert()` path behavior marking `doc.metadata["legacy"] = True`
- validate with existing route-level parse timeout and invalid STEP tests

## Validation Plan

1. `python3 -m py_compile src/core/document_pipeline.py src/api/v1/analyze.py tests/unit/test_document_pipeline.py`
2. `.venv311/bin/flake8 src/core/document_pipeline.py src/api/v1/analyze.py tests/unit/test_document_pipeline.py`
3. `.venv311/bin/python -m pytest -q tests/unit/test_document_pipeline.py tests/unit/test_parse_timeout.py tests/unit/test_step_parse_failure.py tests/integration/test_analyze_quality_pipeline.py tests/integration/test_analyze_process_pipeline.py tests/integration/test_analyze_manufacturing_summary.py tests/integration/test_analyze_vector_pipeline.py tests/test_api_integration.py`
