# DEV_TRAINING_DXF_LABEL_MANIFEST_CLEANED_20260123

## Summary
- Cleaned the training DXF label manifest by merging low-frequency labels into `other`.
- Targeted a minimum label count of 3 to reduce class sparsity.

## Output
- Script: `scripts/clean_dxf_label_manifest.py`
- Cleaned manifest: `reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_CLEANED_20260123.csv`

## Notes
- 47 labels collapsed to 12 (including `other`).
- `other` contains 72 samples after merging.
