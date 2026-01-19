# DEV_MECH_KNOWLEDGE_4000CAD_AUTO_LABEL_ATTEMPT_20260119

## Summary
Attempted to auto-label the 27 unlabeled DWG files using DXF text extraction
and knowledge rules. No text hints matched available rules.

## Steps
- `python3 scripts/auto_label_unlabeled_dxf.py --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf`

## Results
- Unlabeled rows: 27
- Auto-labeled rows: 0
- Output: `reports/MECH_4000_DWG_UNLABELED_LABELS_TEMPLATE_20260119.csv`

## Notes
- These files require manual labeling or additional metadata (e.g., BOM/parts list)
  to infer part names.
