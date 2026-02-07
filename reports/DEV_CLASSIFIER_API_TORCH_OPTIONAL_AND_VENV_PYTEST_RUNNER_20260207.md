# DEV_CLASSIFIER_API_TORCH_OPTIONAL_AND_VENV_PYTEST_RUNNER_20260207

## Summary

Fixed local/unit testing stability by:

1. Making `src/inference/classifier_api.py` importable when `torch` is not installed (torch is optional in this repo).
2. Ensuring `scripts/test_with_local_api.sh` prefers `.venv/bin/pytest` by default so tests run with the project venv dependency set (avoids mismatched global Python site-packages).

This removes hard failures during test collection and makes `make test-unit` reliable in environments without `torch`.

## Changes

### 1) Torch Optional Import for Classifier API

Updated `src/inference/classifier_api.py`:

- Wrapped `torch` import in a `try/except` and introduced `TORCH_AVAILABLE`.
- Added safe fallbacks for `DEVICE` when `torch` is missing.
- Guarded model warmup and lifespan startup:
  - Skip `classifier.load()` and `_warmup_model()` when `TORCH_AVAILABLE` is false.
- Added clear error messages if inference endpoints are called without `torch`.

### 2) Tiered Test Runner Uses Venv Pytest by Default

Updated `scripts/test_with_local_api.sh`:

- Added `default_pytest_bin()` to prefer `.venv/bin/pytest` if present.
- This prevents accidental use of a globally installed `pytest` (and mismatched FastAPI/Starlette versions).

## Validation

Executed using project venv:

```bash
# Unit suite
.venv/bin/pytest -q tests/unit

# Integration suite
.venv/bin/pytest -q tests/integration

# Script checks
bash -n scripts/test_with_local_api.sh
bash scripts/test_with_local_api.sh --help
```

Results:

- Unit: `6993 passed, 26 skipped`
- Integration: `88 passed, 10 skipped`
- Script: syntax OK, help output OK

## Notes

- This change is behavior-preserving for the inference server when `torch` is available.
- When `torch` is missing, the module remains importable (so tests can patch `classifier.load/predict`) but real inference is disabled by design.
