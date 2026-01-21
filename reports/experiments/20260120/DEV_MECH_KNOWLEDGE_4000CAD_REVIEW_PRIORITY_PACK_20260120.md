# DEV_MECH_KNOWLEDGE_4000CAD_REVIEW_PRIORITY_PACK_20260120

## Summary
Built a standalone HTML review pack for the priority DXF conflict samples,
including preview images and a compact CSV for manual triage.

## Command
```
./.venv-graph/bin/python scripts/build_review_priority_pack.py \
  --input reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_PRIORITY_WITH_PREVIEWS_20260120.csv \
  --output-dir reports/experiments/20260120/MECH_4000_DWG_REVIEW_PRIORITY_PACK_20260120
```

## Outputs
- Pack directory: `reports/experiments/20260120/MECH_4000_DWG_REVIEW_PRIORITY_PACK_20260120`
- HTML index: `reports/experiments/20260120/MECH_4000_DWG_REVIEW_PRIORITY_PACK_20260120/index.html`
- CSV: `reports/experiments/20260120/MECH_4000_DWG_REVIEW_PRIORITY_PACK_20260120/review_priority_pack.csv`
- Previews: `reports/experiments/20260120/MECH_4000_DWG_REVIEW_PRIORITY_PACK_20260120/previews`

## Notes
- Preview images are rendered or copied per item; `preview_pack_path` is
  stored in the pack CSV for convenient linking.
