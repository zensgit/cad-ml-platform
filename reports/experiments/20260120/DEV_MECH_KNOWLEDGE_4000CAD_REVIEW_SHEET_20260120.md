# DEV_MECH_KNOWLEDGE_4000CAD_REVIEW_SHEET_20260120

## Summary
Generated a 200-sample DXF review sheet with extracted text, dimensions, and
best-effort title-block fields, plus 20 preview PNGs to support manual
verification.

## Command
```
./.venv-graph/bin/python scripts/generate_dxf_review_sheet.py \
  --dxf-dir "/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf" \
  --manifest reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv \
  --output-csv reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_20260120.csv \
  --preview-dir reports/experiments/20260120/MECH_4000_DWG_REVIEW_PREVIEWS_20260120 \
  --preview-count 20 \
  --sample-size 200
```

## Outputs
- Review CSV: `reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_20260120.csv`
- Instructions: `reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_INSTRUCTIONS_20260120.md`
- Preview images: `reports/experiments/20260120/MECH_4000_DWG_REVIEW_PREVIEWS_20260120`
- Script: `scripts/generate_dxf_review_sheet.py`

## Snapshot Metrics
- Samples: 200
- Text sample non-empty: 122
- Normalized text non-empty: 122
- Title block part name extracted: 4
- Title block drawing number extracted: 0

## Notes
- Some DXF exports store glyphs as `\M+` codes; title-block fields may require
  manual fill or OCR-driven fallback.
- The review sheet includes `review_*` columns to override extracted title
  block metadata and labels.
