# OCR Review Pack CI Validation 2026-03-08

## Scope

This change wires `scripts/export_ocr_review_pack.py` into
`.github/workflows/evaluation-report.yml` as an optional CI artifact.

The goal is to move OCR review-pack generation from a standalone local script
to a CI-visible review artifact with summary lines and PR comment visibility.

## Added Workflow Surface

- `OCR_REVIEW_PACK_ENABLE`
- `OCR_REVIEW_PACK_INPUT`
- `OCR_REVIEW_PACK_OUTPUT_CSV`
- `OCR_REVIEW_PACK_OUTPUT_JSON`
- `OCR_REVIEW_PACK_TOP_K`
- `OCR_REVIEW_PACK_INCLUDE_READY`
- `Build OCR review pack (optional)`
- `Upload OCR review pack`

## Validation

```bash
python3 -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/evaluation-report.yml').read_text())"

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Expected Result

- workflow YAML remains valid
- workflow regression tests verify OCR review-pack envs, step, artifact upload,
  job summary lines, and PR comment contract
