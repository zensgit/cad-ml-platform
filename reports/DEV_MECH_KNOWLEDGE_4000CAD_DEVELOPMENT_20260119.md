# DEV_MECH_KNOWLEDGE_4000CAD_DEVELOPMENT_20260119

## Summary
- Extended DWG manifest/conversion tooling to support recursive, multi-directory inputs.
- Built a 4000CAD manifest and converted DWG to DXF for graph2d training.
- Added English synonym mappings for new labels and refreshed geometry rules.
- Enabled layout-aware OCR-assisted auto-labeling with a high-confidence filter (>= 0.7)
  and installed PaddleOCR/PaddlePaddle locally.
- Added DXF INSERT attribute text extraction for title-block fields.
- Trained a graph2d checkpoint on the filtered DXF dataset.

## Dataset Sources
- `/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸/蜗杆与箱体`
- `/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸/机械零件CAD图纸`

## Manifest + Conversion
- Manifest: `reports/MECH_4000_DWG_LABEL_MANIFEST_20260119.csv` (50 rows, 38 labeled)
- DXF output: `/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf`
- Conversion log: `reports/MECH_4000_DWG_TO_DXF_LOG_20260119.csv`
- OCR note: PaddleOCR runs against paperspace + model layouts, but no text was
  extracted from the rendered DXFs in this slice (no additional labels gained).

## Synonyms + Rules
- Synonyms: `data/knowledge/label_synonyms_template.json` (80 labels)
- Rules updated: `data/knowledge/geometry_rules.json` (no new rules; synonyms check)
- Script used: `python3 scripts/build_geometry_rules_from_manifest.py --manifest reports/MECH_4000_DWG_LABEL_MANIFEST_20260119.csv --synonyms-json data/knowledge/label_synonyms_template.json`

## Training
- Command: `./.venv-graph/bin/python scripts/train_2d_graph.py --manifest reports/MECH_4000_DWG_LABEL_MANIFEST_20260119.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf`
- Checkpoint: `models/graph2d_latest.pth`
- Epoch results:
  - Epoch 1/3 loss=27.0995 val_acc=0.000
  - Epoch 2/3 loss=21.0865 val_acc=0.000
  - Epoch 3/3 loss=5.8363 val_acc=0.000
