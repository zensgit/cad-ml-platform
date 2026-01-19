# DEV_MECH_KNOWLEDGE_4000CAD_AUTO_LABEL_ATTEMPT_20260119

## Summary
Attempted to auto-label the 27 unlabeled DWG files using DXF text extraction,
GBK decode for `\\M+` MTEXT sequences, and knowledge rules. 10 files received
auto labels (rule or heuristic matches).

## Steps
- `python3 scripts/auto_label_unlabeled_dxf.py --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf`

## Results
- Unlabeled rows: 27
- Auto-labeled rows: 10
- Output: `reports/MECH_4000_DWG_UNLABELED_LABELS_TEMPLATE_20260119.csv`

## Notes
- Heuristic labels are based on decoded title-block text and should be verified.
- Remaining unlabeled files still require manual labeling or extra metadata.
