# DEV_MECH_KNOWLEDGE_4000CAD_OCR_AUTO_LABEL_20260121

## Summary
- Executed OCR-enabled auto-labeling against the 4000CAD DXF set using PaddleOCR.
- OCR text extraction produced no usable text on this batch; all labels came from DXF text and filename heuristics.
- Generated a delta file comparing OCR run output to the prior auto-label pass.

## Inputs
- Unlabeled template: `reports/MECH_4000_DWG_UNLABELED_LABELS_TEMPLATE_20260119.csv`
- DXF source: `/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf`
- Synonyms: `data/knowledge/label_synonyms_template.json`

## Outputs
- OCR auto labels: `reports/experiments/20260121/MECH_4000_DWG_UNLABELED_AUTO_OCR_20260121.csv`
- OCR deltas: `reports/experiments/20260121/MECH_4000_DWG_UNLABELED_OCR_DELTAS_20260121.csv`

## Key Stats
- Rows processed: 27
- Labeled rows: 27
- OCR-used rows: 0
- Top labels: 练习零件图(9), 旋转组件(2), 挡板(2), 盖(2), 轴承(2)
- Delta rows vs previous auto-label: 10 (label_en backfilled via synonyms)

## Commands
- `./.venv-graph/bin/python scripts/auto_label_unlabeled_dxf.py --input-csv reports/MECH_4000_DWG_UNLABELED_LABELS_TEMPLATE_20260119.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --output-csv reports/experiments/20260121/MECH_4000_DWG_UNLABELED_AUTO_OCR_20260121.csv --enable-ocr`

## Notes
- OCR initialization logged `dxf_render_extents_invalid` warnings on some files; no extracted text was returned.
