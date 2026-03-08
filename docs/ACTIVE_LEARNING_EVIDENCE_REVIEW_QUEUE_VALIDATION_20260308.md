# Active Learning Evidence Review Queue Validation

Date: 2026-03-08

## Scope

- Surface assistant structured evidence from `score_breakdown` into the active-learning review queue payload.
- Preserve the same evidence context in exported review/training JSONL.
- Keep the upstream analyze payload contract unchanged.

## Implemented Behavior

- `ActiveLearningSample` now derives reviewer-facing evidence fields from `score_breakdown`:
  - `evidence_count`
  - `evidence_sources`
  - `evidence_summary`
  - `evidence`
- Evidence is built from existing hybrid-classifier context when present:
  - `source_contributions`
  - `hybrid_explanation.summary`
  - `hybrid_rejection`
  - `decision_path`
  - `fusion_metadata`
- `GET /api/v1/active-learning/pending` exposes those fields through the existing sample response model.
- `POST /api/v1/active-learning/export` writes the same evidence fields into exported JSONL rows.
- File-backed samples loaded from `samples.jsonl` re-derive evidence from legacy `score_breakdown` content.

## Validation Commands

```bash
python3 -m black src/core/active_learning.py tests/test_active_learning_api.py tests/unit/test_active_learning_export_context.py tests/unit/test_active_learning_loop.py
pytest tests/test_active_learning_api.py tests/unit/test_active_learning_export_context.py tests/unit/test_active_learning_loop.py tests/integration/test_analyze_dxf_active_learning_context.py -v
```

## Validation Results

- `black`: reformatted `src/core/active_learning.py` and `tests/test_active_learning_api.py`; no errors.
- `pytest`: `20 passed` in `1.10s`.
- Warnings observed only from existing third-party deprecations in `starlette/httpx/ezdxf`; no active-learning failures.

## Coverage Notes

- API queue coverage:
  - empty evidence defaults
  - structured evidence exposed in `/pending`
  - structured evidence persisted into `/export`
- Core/file-store coverage:
  - evidence derived at flag time
  - evidence exported with `score_breakdown`
  - evidence re-derived when loading existing `samples.jsonl`
