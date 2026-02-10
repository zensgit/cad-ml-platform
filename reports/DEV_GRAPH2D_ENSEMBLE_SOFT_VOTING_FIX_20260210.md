# DEV_GRAPH2D_ENSEMBLE_SOFT_VOTING_FIX_20260210

## Goal
Make Graph2D ensemble `voting=soft` actually average **per-class probabilities** across
models (instead of reconstructing fake distributions from `{label, confidence}`), so:
- ensemble predictions are mathematically meaningful
- we can compute `top2_confidence` + `margin` for guardrails (`GRAPH2D_MIN_MARGIN`)

## Changes
- Updated `src/ml/vision_2d.py`:
  - `Graph2DClassifier` now exposes an internal `_predict_probs(...)` helper that returns
    the full probability vector (kept internal; public payload stays small).
  - `EnsembleGraph2DClassifier` soft voting now:
    - aligns class indices by label names (requires consistent label sets)
    - averages probability vectors across models
    - outputs `top2_confidence` and `margin`
    - falls back to hard voting when label maps mismatch.

## Compatibility
- Public `Graph2DClassifier.predict_from_bytes(...)` response remains compatible.
- Ensemble response adds:
  - `top2_confidence`, `margin`, `label_map_size`
  - `label_map_mismatch` and `voting=hard_fallback_label_map_mismatch` when fallback occurs.

## Verification
- Unit tests added:
  - `tests/unit/test_vision_2d_ensemble_voting.py`
    - verifies probability averaging, margin calculation
    - verifies hard-vote fallback on label-map mismatch

Example:
```bash
python3 -m pytest tests/unit/test_vision_2d_ensemble_voting.py -q
```

