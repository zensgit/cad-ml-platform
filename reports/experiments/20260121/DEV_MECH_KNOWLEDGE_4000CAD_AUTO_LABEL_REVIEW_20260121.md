# DEV_MECH_KNOWLEDGE_4000CAD_AUTO_LABEL_REVIEW_20260121

## Summary
- Ran auto-labeling on the 27-row unlabeled DWG template against the 4000CAD DXF set.
- Generated a focused review sheet for the unlabeled subset, ran Graph2D auto-review, and built a priority HTML pack for conflicts.

## Inputs
- Unlabeled template: `reports/MECH_4000_DWG_UNLABELED_LABELS_TEMPLATE_20260119.csv`
- DXF source: `/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf`
- Graph2D model: `models/graph2d_merged_latest.pth`

## Outputs
- Auto labels: `reports/experiments/20260121/MECH_4000_DWG_UNLABELED_AUTO_20260121.csv`
- Review sheet: `reports/experiments/20260121/MECH_4000_DWG_REVIEW_SHEET_20260121.csv`
- Review previews: `reports/experiments/20260121/MECH_4000_DWG_REVIEW_PREVIEWS_20260121/`
- Auto review: `reports/experiments/20260121/MECH_4000_DWG_REVIEW_SHEET_AUTO_20260121.csv`
- Conflicts: `reports/experiments/20260121/MECH_4000_DWG_REVIEW_SHEET_CONFLICTS_20260121.csv`
- Conflict summary: `reports/experiments/20260121/MECH_4000_DWG_REVIEW_SHEET_CONFLICTS_SUMMARY_20260121.csv`
- Priority pack: `reports/experiments/20260121/MECH_4000_DWG_REVIEW_PRIORITY_PACK_20260121/`

## Key Stats
- Auto-labeled rows: 27 (26 with labels after thresholding)
- Review sheet rows: 27
- Conflicts after Graph2D auto-review: 16

## Commands
- `python3 scripts/auto_label_unlabeled_dxf.py --input-csv reports/MECH_4000_DWG_UNLABELED_LABELS_TEMPLATE_20260119.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --output-csv reports/experiments/20260121/MECH_4000_DWG_UNLABELED_AUTO_20260121.csv --enable-ocr --min-confidence 0.7`
- `python3 scripts/generate_dxf_review_sheet.py --dxf-dir reports/experiments/20260121/dxf_unlabeled --manifest reports/experiments/20260121/MECH_4000_DWG_UNLABELED_AUTO_20260121.csv --output-csv reports/experiments/20260121/MECH_4000_DWG_REVIEW_SHEET_20260121.csv --sample-size 27 --seed 21 --preview-dir reports/experiments/20260121/MECH_4000_DWG_REVIEW_PREVIEWS_20260121 --preview-count 10 --render-dpi 200 --render-size 1024`
- `./.venv-graph/bin/python scripts/auto_review_dxf_sheet.py --input reports/experiments/20260121/MECH_4000_DWG_REVIEW_SHEET_20260121.csv --output reports/experiments/20260121/MECH_4000_DWG_REVIEW_SHEET_AUTO_20260121.csv --conflicts reports/experiments/20260121/MECH_4000_DWG_REVIEW_SHEET_CONFLICTS_20260121.csv --graph2d-model models/graph2d_merged_latest.pth --confidence-threshold 0.2`
- `python3 scripts/build_review_priority_pack.py --input reports/experiments/20260121/MECH_4000_DWG_REVIEW_SHEET_CONFLICTS_20260121.csv --output-dir reports/experiments/20260121/MECH_4000_DWG_REVIEW_PRIORITY_PACK_20260121 --render-dpi 200 --render-size 1024`

## Notes
- OCR was requested but PaddleOCR was unavailable in the current environment; auto-labeling fell back to DXF text extraction only.
- Manual decision application is documented in `DEV_MECH_KNOWLEDGE_4000CAD_AUTO_LABEL_REVIEW_APPLIED_20260121.md`.
