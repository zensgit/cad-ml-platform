# DEV_MECH_KNOWLEDGE_4000CAD_AUTO_LABEL_ATTEMPT_20260119

## Summary
Auto-labeled all 27 previously unlabeled DWG files using decoded DXF text,
knowledge rules, and filename fallbacks. Numeric-only filenames were tagged
as generic practice drawings.

## Steps
- `python3 scripts/auto_label_unlabeled_dxf.py --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf`

## Results
- Unlabeled rows: 27
- Auto-labeled rows: 27
- Output: `reports/MECH_4000_DWG_UNLABELED_LABELS_TEMPLATE_20260119.csv`

## Notes
- Heuristic and filename-based labels (e.g., `练习零件图`, `直推`) should be verified.
- Rule-based labels cover items like `差动机构`, `蜗杆`, `轴承座`.
