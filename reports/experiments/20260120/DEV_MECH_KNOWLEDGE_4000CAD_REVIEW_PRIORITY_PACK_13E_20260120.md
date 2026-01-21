# DEV_MECH_KNOWLEDGE_4000CAD_REVIEW_PRIORITY_PACK_13E_20260120

## Summary
Built an HTML review pack for the final 13 conflict samples with preview
renders for manual review.

## Command
```
python3 scripts/build_review_priority_pack.py \
  --input reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_PRIORITY_13E_WITH_PREVIEWS_20260120.csv \
  --output-dir reports/experiments/20260120/MECH_4000_DWG_REVIEW_PRIORITY_PACK_13E_20260120
```

## Outputs
- Pack directory: `reports/experiments/20260120/MECH_4000_DWG_REVIEW_PRIORITY_PACK_13E_20260120`
- HTML review page: `reports/experiments/20260120/MECH_4000_DWG_REVIEW_PRIORITY_PACK_13E_20260120/index.html`
- Pack CSV: `reports/experiments/20260120/MECH_4000_DWG_REVIEW_PRIORITY_PACK_13E_20260120/review_priority_pack.csv`
- Preview images: `reports/experiments/20260120/MECH_4000_DWG_REVIEW_PRIORITY_PACK_13E_20260120/previews/*.png`

## Notes
- DXF rendering emits warnings for some MTEXT/DIMASSOC entries; renders still
  complete.
