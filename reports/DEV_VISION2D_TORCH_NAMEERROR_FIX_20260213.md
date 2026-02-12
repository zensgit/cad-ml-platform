# DEV_VISION2D_TORCH_NAMEERROR_FIX_20260213

## Goal

Fix the `CI Tiered Tests` unit-tier regression:

- `tests/unit/test_vision_2d_ensemble_voting.py::TestGraph2DClassifier::test_predict_probs_empty_input`
- Failure on GitHub Actions: `NameError: name 'torch' is not defined`

## Root Cause

`src/ml/vision_2d.py` uses optional torch import with `HAS_TORCH` flag.
When torch import fails, the module did not define `torch` at all.

Some tests monkeypatch `HAS_TORCH=True` to exercise code paths with stubs.
If a default model path exists, `Graph2DClassifier.__init__` enters `_load_model()`,
which referenced `torch.load(...)` and raised `NameError`.

## Changes

- `src/ml/vision_2d.py`
  - In torch import fallback branch, define `torch = None` explicitly.
  - Keep runtime gating behavior based on `HAS_TORCH` for test compatibility.
  - Add defensive guard in `_load_model()`:
    - return early when `HAS_TORCH` is false **or** `torch is None`.

This is a minimal compatibility fix: no model output logic or API payload contract changed.

## Validation

- Targeted regression suite:
  - `.venv/bin/python -m pytest tests/unit/test_vision_2d_ensemble_voting.py -v`
  - Result: `26 passed`
- Core project fast gate:
  - `make validate-core-fast`
  - Result: passed

