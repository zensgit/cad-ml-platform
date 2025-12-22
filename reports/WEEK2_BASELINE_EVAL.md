#!/usr/bin/env markdown
# Week 2 Report: Baseline Evaluation

## Summary
- Added baseline evaluation guidance and ensured the script does not hard-fail
  when assembly modules are unavailable.

## Changes
- Added `docs/BASELINE_EVALUATION.md` and linked in `README.md`.
- `scripts/run_baseline_evaluation.py` now skips gracefully if assembly modules
  are missing.

## Tests
- `make test-baseline` (skipped; assembly module not available).

## Verification
- Expected output file when module is available:
  - `evaluation_results/baseline_YYYYMMDD_HHMMSS.json`
