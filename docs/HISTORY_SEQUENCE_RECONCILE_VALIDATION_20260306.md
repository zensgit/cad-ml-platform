# History Sequence Reconcile Validation

## Scope

Branch: `feat/history-sequence-reconcile`

Commits included:
- `a6d562f` `feat: reconcile history sequence tooling with main`
- `7b195b0` `feat: validate history sequence eval records`
- `975444d` `feat: add history sequence tooling scripts`

## What Changed

- Reconciled history-sequence utility APIs with current `main`.
- Preserved `HPSketchSequenceDataset` and added self-supervised sequence dataset helpers.
- Added tooling scripts:
  - `scripts/build_history_sequence_prototypes.py`
  - `scripts/eval_history_sequence_classifier.py`
  - `scripts/tune_history_sequence_weights.py`
- Extended eval-history validation to accept `history_sequence` records.

## Key Files

- `src/ml/history_sequence_tools.py`
- `src/ml/train/hpsketch_dataset.py`
- `scripts/build_history_sequence_prototypes.py`
- `scripts/eval_history_sequence_classifier.py`
- `scripts/tune_history_sequence_weights.py`
- `scripts/validate_eval_history.py`
- `docs/eval_history.schema.json`

## Validation

Commands:

```bash
python3 -m pytest -q \
  tests/unit/test_history_sequence_tools.py \
  tests/unit/test_hpsketch_dataset.py \
  tests/unit/test_history_sequence_classifier.py \
  tests/unit/test_sequence_encoder.py \
  tests/unit/test_history_sequence_scripts.py \
  tests/unit/test_build_history_sequence_prototypes.py \
  tests/unit/test_eval_history_sequence_classifier.py \
  tests/unit/test_tune_history_sequence_weights.py \
  tests/unit/test_validate_eval_history_history_sequence.py

flake8 \
  src/ml/history_sequence_tools.py \
  src/ml/train/hpsketch_dataset.py \
  scripts/build_history_sequence_prototypes.py \
  scripts/eval_history_sequence_classifier.py \
  scripts/tune_history_sequence_weights.py \
  scripts/validate_eval_history.py \
  tests/unit/test_history_sequence_tools.py \
  tests/unit/test_hpsketch_dataset.py \
  tests/unit/test_history_sequence_classifier.py \
  tests/unit/test_sequence_encoder.py \
  tests/unit/test_history_sequence_scripts.py \
  tests/unit/test_build_history_sequence_prototypes.py \
  tests/unit/test_eval_history_sequence_classifier.py \
  tests/unit/test_tune_history_sequence_weights.py \
  tests/unit/test_validate_eval_history_history_sequence.py \
  --max-line-length=100

python3 -m py_compile \
  src/ml/history_sequence_tools.py \
  src/ml/train/hpsketch_dataset.py \
  scripts/build_history_sequence_prototypes.py \
  scripts/eval_history_sequence_classifier.py \
  scripts/tune_history_sequence_weights.py \
  scripts/validate_eval_history.py
```

Results:

- `33 passed`
- `flake8` passed
- `py_compile` passed

## Risks

- History tooling introduces new CLI entrypoints and artifact formats.
- `feat/graph2d-eval-history` depends on this branch because `eval_with_history.sh` now calls the history tooling scripts.
