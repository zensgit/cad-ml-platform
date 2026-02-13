# DEV_DXF_TITLEBLOCK_COVERAGE_EVAL_20260214

## Goal

Establish a measurable baseline for TitleBlock extraction/classification on real DXF drawings (without manual review), and quantify how often titleblock-derived labels agree with filename weak labels.

## Changes

### 1) New Offline Evaluator Script

File: `scripts/eval_titleblock_on_dxf_dir.py`

- Scans a DXF directory and runs:
  - `TitleBlockClassifier` (titleblock text extraction + label match)
  - `FilenameClassifier` (weak label from filename)
- Writes local-only artifacts:
  - `summary.json` (aggregate counters + agreement rates)
  - `predictions.csv` (per-file predictions)
  - `errors.csv` (parse/read failures)

### 2) Unit Tests

File: `tests/unit/test_titleblock_extractor_and_classifier.py`

- Covers:
  - INSERT + ATTRIB tag path (tests part-name normalization for spec suffixes like `DN####`)
  - INSERT `virtual_entities()` path (titleblock text embedded inside blocks)

## Validation

### Static Checks

- `.venv/bin/python -m py_compile scripts/eval_titleblock_on_dxf_dir.py src/ml/titleblock_extractor.py` (passed)

### Unit Tests

- `.venv/bin/pytest tests/unit/test_titleblock_extractor_and_classifier.py -q` (passed)

### Local Coverage Run (Training Drawings)

Command:

```bash
/usr/bin/time -p .venv/bin/python scripts/eval_titleblock_on_dxf_dir.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --recursive \
  --max-files 200
```

Observed:

- Titleblock vs filename agreement (weak-label based):
  - `both_present=110`
  - `agree=108`
  - `agree_rate=0.981818`
  - `strict_truth_agree_rate=0.981818` (filename conf >= 0.8)
- Artifacts: `/tmp/titleblock_eval_20260214_010412`
- Timing: `real 17.85s`

## Notes / Caveats

- Agreement is computed against filename weak labels, not human-verified ground truth.
- This evaluator is designed for local iteration; artifacts contain local paths and should not be committed.

