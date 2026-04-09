# Assistant Evidence Report CI Validation

## Scope

- Branch: `feat/assistant-evidence-report-ci`
- Base: `feat/assistant-evidence-report-export`
- Date: `2026-03-08`

## Delivered

- `evaluation-report.yml` now supports optional assistant evidence report export
- new workflow_dispatch inputs:
  - `assistant_evidence_report_enable`
  - `assistant_evidence_report_input`
- new env controls:
  - `ASSISTANT_EVIDENCE_REPORT_ENABLE`
  - `ASSISTANT_EVIDENCE_REPORT_INPUT`
  - `ASSISTANT_EVIDENCE_REPORT_OUTPUT_CSV`
  - `ASSISTANT_EVIDENCE_REPORT_OUTPUT_JSON`
  - `ASSISTANT_EVIDENCE_REPORT_TOP_K`
- new step:
  - `Build assistant evidence report (optional)`
- new artifact upload:
  - `Upload assistant evidence report`
- job summary additions:
  - assistant evidence input / records / evidence items
  - average evidence count
  - evidence / decision_path / source signal coverage
  - top record kinds / evidence types / structured sources / missing fields
- PR comment additions:
  - `Assistant Evidence Report`
  - `Assistant Evidence Insights`

## Merge Cleanup Included

This branch also fixes two workflow integration issues carried forward from earlier stacked merges:

- `benchmark_scorecard_enable` dispatch input now has explicit `required/default`
- benchmark and OCR summary shell blocks now close cleanly before the next optional section

## Validation

Commands:

```bash
python3 -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/evaluation-report.yml').read_text())"
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Expected result:

- workflow YAML parses successfully
- regression test passes and asserts assistant evidence report wiring plus existing benchmark/OCR sections
