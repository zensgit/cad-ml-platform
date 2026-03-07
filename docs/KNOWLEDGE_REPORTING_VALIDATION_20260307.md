# Knowledge Reporting Validation 2026-03-07

## Goal
Expose engineering knowledge outputs in evaluation and review/reporting artifacts so product, review, and CI surfaces can see knowledge hits and standard candidates instead of only conflict counts.

## Scope
Updated:
- `scripts/eval_hybrid_dxf_manifest.py`
- `scripts/batch_analyze_dxf_local.py`
- `scripts/export_hybrid_rejection_review_pack.py`
- `.github/workflows/evaluation-report.yml`
- unit tests for eval/review/workflow coverage

## Delivered
- Manifest evaluation rows now include raw JSON and derived tokens for:
  - `knowledge_checks`
  - `violations`
  - `standards_candidates`
  - `knowledge_hints`
- Manifest summary now includes:
  - `knowledge_signals.rows_with_checks`
  - `knowledge_signals.rows_with_violations`
  - `knowledge_signals.rows_with_standards_candidates`
  - `knowledge_signals.rows_with_hints`
  - top categories / standard types / hint labels
- Local batch DXF output now persists the same knowledge fields into CSV/summary JSON.
- Hybrid rejection review pack now summarizes:
  - `knowledge_check_row_count`
  - `standards_candidate_row_count`
  - `top_knowledge_check_categories`
  - `top_standard_candidate_types`
  - `top_knowledge_hint_labels`
- Evaluation workflow now exports these values into step outputs and GitHub job summary.
- PR comment insight string now includes knowledge category and standard candidate summaries.

## Validation
Commands run:

```bash
python3 -m py_compile \
  scripts/eval_hybrid_dxf_manifest.py \
  scripts/batch_analyze_dxf_local.py \
  scripts/export_hybrid_rejection_review_pack.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  tests/unit/test_export_hybrid_rejection_review_pack.py \
  tests/unit/test_export_hybrid_rejection_review_pack_context.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_batch_analyze_dxf_local_knowledge_context.py

flake8 \
  scripts/eval_hybrid_dxf_manifest.py \
  scripts/batch_analyze_dxf_local.py \
  scripts/export_hybrid_rejection_review_pack.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  tests/unit/test_export_hybrid_rejection_review_pack.py \
  tests/unit/test_export_hybrid_rejection_review_pack_context.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_batch_analyze_dxf_local_knowledge_context.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  tests/unit/test_export_hybrid_rejection_review_pack.py \
  tests/unit/test_export_hybrid_rejection_review_pack_context.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_batch_analyze_dxf_local_knowledge_context.py
```

Results:
- `py_compile`: pass
- `flake8`: pass
- `pytest`: `12 passed`

## Notes
- This change is additive. It does not alter final classification decisions.
- The reporting code is generic over `knowledge_checks`/`standards_candidates` content, so when richer knowledge extraction lands, the reporting path will surface new categories automatically.
