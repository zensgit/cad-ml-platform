# Eval History Coarse Schema Validation

Date: `2026-03-07`

## Goal

Extend evaluation-history validation so history-sequence records can carry the
new coarse-label metrics and mismatch summaries introduced by the batch
history-sequence evaluation flow.

## Changes

- Extended `scripts/validate_eval_history.py` to validate these optional
  `history_metrics` fields:
  - `coarse_accuracy_on_ok`
  - `coarse_accuracy_overall`
  - `coarse_macro_f1_on_ok`
  - `coarse_macro_f1_overall`
- Added structural validation for:
  - `exact_top_mismatches`
  - `coarse_top_mismatches`
- Extended `docs/eval_history.schema.json` to describe the same fields
  explicitly.
- Added validator tests for:
  - accepting well-formed coarse history metrics
  - rejecting malformed mismatch rows

## Files

- `scripts/validate_eval_history.py`
- `docs/eval_history.schema.json`
- `tests/unit/test_validate_eval_history_history_sequence.py`

## Validation

```bash
python3 -m py_compile \
  scripts/validate_eval_history.py \
  tests/unit/test_validate_eval_history_history_sequence.py

flake8 \
  scripts/validate_eval_history.py \
  tests/unit/test_validate_eval_history_history_sequence.py \
  --max-line-length=100

pytest -q tests/unit/test_validate_eval_history_history_sequence.py
```

Result:

- `2 passed`
- `1 warning`

## Notes

- This change is additive. Existing history records remain valid.
- The validator now catches malformed mismatch payloads before they enter
  evaluation history or CI reporting flows.
