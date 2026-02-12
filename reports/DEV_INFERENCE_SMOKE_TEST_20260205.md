# DEV_INFERENCE_SMOKE_TEST_20260205

## Summary
Ran a local smoke test using a minimal DXF file to confirm feature extraction
and V16 inference path execute end-to-end with local model weights.

## Verification
- `python3 - <<'PY'`
  - Creates a minimal DXF (line + circle)
  - Calls `extract_features_v6`
  - Calls `V16Classifier.predict`

## Result
- Feature vector produced: `(48,)` with non-zero sum.
- V16 prediction returned successfully with category + confidence.

## Notes
- Requires local model weights in `models/` and `ezdxf` installed.
