# History Sequence Coarse Eval Validation

Date: `2026-03-07`

## Goal

Bring `scripts/eval_history_sequence_classifier.py` onto the same coarse/fine
evaluation contract already used by DXF and hybrid evaluation flows.

## Changes

- Added coarse-label normalization to history-sequence batch evaluation.
- Added coarse-level metrics to `summary.json`:
  - `coarse_accuracy_on_ok`
  - `coarse_accuracy_overall`
  - `coarse_macro_f1_on_ok`
  - `coarse_macro_f1_overall`
- Added mismatch summaries:
  - `exact_top_mismatches`
  - `coarse_top_mismatches`
- Extended `results.csv` rows with:
  - `expected_coarse_label`
  - `predicted_coarse_label`
  - `coarse_ok`
- Updated real-data command documentation to call out the new summary keys.

## Files

- `scripts/eval_history_sequence_classifier.py`
- `tests/unit/test_eval_history_sequence_classifier.py`
- `tests/unit/test_history_sequence_scripts.py`
- `docs/REAL_DATA_VALIDATION_COMMANDS_20260306.md`

## Validation

### Static checks

```bash
python3 -m py_compile \
  scripts/eval_history_sequence_classifier.py \
  tests/unit/test_eval_history_sequence_classifier.py \
  tests/unit/test_history_sequence_scripts.py

flake8 \
  scripts/eval_history_sequence_classifier.py \
  tests/unit/test_eval_history_sequence_classifier.py \
  tests/unit/test_history_sequence_scripts.py \
  --max-line-length=100
```

Result:

- passed

### Unit tests

Run inside the existing `cad-ml-brep-m4` micromamba environment because the
history tests require `h5py`.

```bash
~/.micromamba/envs/cad-ml-brep-m4/bin/python -m pytest -q \
  tests/unit/test_eval_history_sequence_classifier.py \
  tests/unit/test_history_sequence_scripts.py
```

Result:

- `4 passed`

### Online `.h5` smoke evaluation

```bash
~/.micromamba/envs/cad-ml-brep-m4/bin/python \
  scripts/eval_history_sequence_classifier.py \
  --manifest /tmp/history_eval_manifest_20260307.json \
  --label-source manifest \
  --min-seq-len 4 \
  --output-dir reports/experiments/20260307/history_sequence_online_eval_smoke
```

Manifest content:

- `/private/tmp/cad-ai-example-data-20260307/HPSketch/data/0000/00000007_1.h5`
- label: `轴类`

Smoke result:

- `total=1`
- `ok_count=1`
- `accuracy_overall=1.0`
- `coarse_accuracy_overall=1.0`
- `coarse_macro_f1_overall=1.0`
- `exact_top_mismatches=[]`
- `coarse_top_mismatches=[]`

## Notes

- The online smoke validates the CLI path and summary contract, not benchmark
  quality.
- This change is deliberately low-conflict: no API contract files or main
  fusion files were touched.
