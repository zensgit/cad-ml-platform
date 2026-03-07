# History Eval Record Coarse Validation

Date: `2026-03-07`

## Goal

Ensure `scripts/eval_with_history.sh` carries the new history-sequence coarse
metrics from `summary.json` into the persisted `reports/eval_history/*_history_sequence.json`
records.

## Changes

- `scripts/eval_with_history.sh`
  - Added env overrides for:
    - `EVAL_HISTORY_REPORT_DIR`
    - `EVAL_HISTORY_OCR_SCRIPT`
    - `EVAL_HISTORY_BUILD_SCRIPT`
    - `EVAL_HISTORY_EVAL_SCRIPT`
    - `EVAL_HISTORY_TUNE_SCRIPT`
  - Added coarse history fields to both `metrics` and `history_metrics`:
    - `coarse_accuracy_on_ok`
    - `coarse_accuracy_overall`
    - `coarse_macro_f1_on_ok`
    - `coarse_macro_f1_overall`
    - `exact_top_mismatches`
    - `coarse_top_mismatches`

- `tests/unit/test_graph2d_eval_helpers.py`
  - Added a script-level test that injects fake OCR/history scripts and
    verifies the generated history eval record contains the new coarse fields.

## Validation

```bash
bash -n scripts/eval_with_history.sh
python3 -m py_compile tests/unit/test_graph2d_eval_helpers.py
flake8 tests/unit/test_graph2d_eval_helpers.py --max-line-length=100
pytest -q tests/unit/test_graph2d_eval_helpers.py
```

Result:

- `4 passed`
- warnings only from optional third-party libs already present in the repo

## Notes

- The env overrides make the shell workflow testable without patching the real
  OCR or history CLI stack.
- This is additive; existing usage keeps the default script paths.
