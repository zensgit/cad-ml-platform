# DEV_MECH_KNOWLEDGE_4000CAD_DEVELOPMENT_20260119

## Summary
- Extended DWG manifest/conversion tooling to support recursive, multi-directory inputs.
- Built a 4000CAD manifest and converted DWG to DXF for graph2d training.
- Added English synonym mappings for new labels and updated geometry rules.
- Trained a graph2d checkpoint on the new DXF dataset.

## Dataset Sources
- `/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸/蜗杆与箱体`
- `/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸/机械零件CAD图纸`

## Manifest + Conversion
- Manifest: `reports/MECH_4000_DWG_LABEL_MANIFEST_20260119.csv` (50 rows)
- DXF output: `/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf`
- Conversion log: `reports/MECH_4000_DWG_TO_DXF_LOG_20260119.csv`

## Synonyms + Rules
- Synonyms: `data/knowledge/label_synonyms_template.json` (68 labels)
- Rules updated: `data/knowledge/geometry_rules.json` (+21 dataset rules, +4 from auto-labeled files)
- Script used: `python3 scripts/build_geometry_rules_from_manifest.py --manifest reports/MECH_4000_DWG_LABEL_MANIFEST_20260119.csv --synonyms-json data/knowledge/label_synonyms_template.json`

## Training
- Command: `./.venv-graph/bin/python scripts/train_2d_graph.py --manifest reports/MECH_4000_DWG_LABEL_MANIFEST_20260119.csv --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf`
- Checkpoint: `models/graph2d_latest.pth`
- Epoch results:
  - Epoch 1/3 loss=3.6102 val_acc=0.200
  - Epoch 2/3 loss=3.2737 val_acc=0.200
  - Epoch 3/3 loss=3.0672 val_acc=0.200
