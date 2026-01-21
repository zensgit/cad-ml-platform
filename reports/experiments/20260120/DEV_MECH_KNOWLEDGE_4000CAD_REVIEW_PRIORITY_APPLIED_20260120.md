# DEV_MECH_KNOWLEDGE_4000CAD_REVIEW_PRIORITY_APPLIED_20260120

## Summary
Applied the 10 priority manual decisions to the review sheets, synced the
merged manifest, rebuilt geometry rules, retrained the merged Graph2D model,
and refreshed the priority review pack.

## Manual Decisions Applied
- 1200型风送式喷雾机 → 型风送式喷雾机
- 基质沥青计量罐 → 基质沥青计量罐
- 前进离合器A1 → 前进离合器
- 轴承外圈强化研磨上下料系统11-2 → 装配图
- 后机架更换桥6.20 → 装配图
- 机械制图77 → 机械制图
- 31001-1 → 装配图
- 4后桥装配 → 装配图
- 4后桥装配 - 副本 → 装配图
- JDB00000 → 装配图

## Updated Artifacts
- Review sheets:
  - `reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_20260120.csv`
  - `reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_AUTO_20260120.csv`
  - `reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_CONFLICTS_20260120.csv`
  - `reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_PRIORITY_20260120.csv`
  - `reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_PRIORITY_WITH_PREVIEWS_20260120.csv`
- Priority pack:
  - `reports/experiments/20260120/MECH_4000_DWG_REVIEW_PRIORITY_PACK_20260120`
- Manifest:
  - `reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv`

## Rules + Training
Commands:
```
python3 scripts/build_geometry_rules_from_manifest.py \
  --synonyms-json data/knowledge/label_synonyms_template.json

./.venv-graph/bin/python scripts/train_2d_graph.py \
  --manifest reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv \
  --dxf-dir "/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf" \
  --output models/graph2d_merged_latest.pth
```

Training result:
- Epoch 1/3 loss=4.2375 val_acc=0.311
- Epoch 2/3 loss=6.1034 val_acc=0.289
- Epoch 3/3 loss=3.5303 val_acc=0.222
- Saved: `models/graph2d_merged_latest.pth`

## Priority Pack Refresh
Command:
```
./.venv-graph/bin/python scripts/build_review_priority_pack.py \
  --input reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_PRIORITY_WITH_PREVIEWS_20260120.csv \
  --output-dir reports/experiments/20260120/MECH_4000_DWG_REVIEW_PRIORITY_PACK_20260120
```
