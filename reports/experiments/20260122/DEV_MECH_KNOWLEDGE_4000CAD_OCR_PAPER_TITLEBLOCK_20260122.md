# DEV_MECH_KNOWLEDGE_4000CAD_OCR_PAPER_TITLEBLOCK_20260122

## Summary
- Ran OCR with paper-space focus and title-block crop to capture template/title metadata.
- OCR extraction returned no text for this batch; labels unchanged from DXF/filename signals.

## Inputs
- Unlabeled template: `reports/MECH_4000_DWG_UNLABELED_LABELS_TEMPLATE_20260119.csv`
- DXF source: `/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf`
- OCR settings: `--ocr-layout paper --ocr-crop-ratio 0.2`

## Outputs
- OCR paper auto labels: `reports/experiments/20260122/MECH_4000_DWG_UNLABELED_AUTO_OCR_PAPER_20260122.csv`

## Key Stats
- Rows processed: 27
- OCR-used rows: 0
- Labels found: 14

## Commands
- `./.venv-graph/bin/python scripts/auto_label_unlabeled_dxf.py --input-csv reports/MECH_4000_DWG_UNLABELED_LABELS_TEMPLATE_20260119.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --output-csv reports/experiments/20260122/MECH_4000_DWG_UNLABELED_AUTO_OCR_PAPER_20260122.csv --enable-ocr --ocr-layout paper --ocr-crop-ratio 0.2`

## Notes
- Multiple `dxf_render_extents_invalid` warnings were logged during rendering.
