# Graph2D Eval Diagnostics Validation

## Scope

Branch: `feat/graph2d-eval-history`

Commit:
- `3632b74` `feat: extend graph2d evaluation diagnostics`

Base branch:
- `feat/history-sequence-reconcile`

## What Changed

- Added low-confidence summary fields to `scripts/diagnose_graph2d_on_dxf_dir.py`.
- Made `scripts/eval_trend.py` explicitly recognize `type=ocr`.
- Extended `scripts/eval_with_history.sh` to optionally run history-sequence build, tune, and eval steps.
- Added a repository-local `.gitignore` for `reports/eval_history/mplcache/`.

## Key Files

- `scripts/diagnose_graph2d_on_dxf_dir.py`
- `scripts/eval_trend.py`
- `scripts/eval_with_history.sh`
- `reports/eval_history/mplcache/.gitignore`
- `tests/unit/test_graph2d_eval_helpers.py`

## Validation

Commands:

```bash
python3 -m pytest -q \
  tests/unit/test_diagnose_graph2d_no_text_no_filename_flags.py \
  tests/unit/test_diagnose_graph2d_manifest_truth.py \
  tests/unit/test_run_graph2d_pipeline_local_diagnose_strict_wiring.py \
  tests/unit/test_graph2d_eval_helpers.py

flake8 \
  scripts/diagnose_graph2d_on_dxf_dir.py \
  scripts/eval_trend.py \
  tests/unit/test_graph2d_eval_helpers.py \
  --max-line-length=100

python3 -m py_compile \
  scripts/diagnose_graph2d_on_dxf_dir.py \
  scripts/eval_trend.py \
  tests/unit/test_graph2d_eval_helpers.py
```

Results:

- `8 passed`
- `flake8` passed
- `py_compile` passed
- `eval_with_history.sh` bash syntax validated from tests via `bash -n`

## Risks

- This branch depends on the history-sequence tooling branch because `eval_with_history.sh` now invokes those scripts.
- `eval_with_history.sh` broadens evaluation scope and may surface missing environment setup in local/CI runs.
