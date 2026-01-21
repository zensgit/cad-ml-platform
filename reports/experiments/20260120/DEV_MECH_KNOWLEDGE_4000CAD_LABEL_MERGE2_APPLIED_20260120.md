# DEV_MECH_KNOWLEDGE_4000CAD_LABEL_MERGE2_APPLIED_20260120

## Summary
Applied a second-stage merge to consolidate low-accuracy labels into `机械制图`,
reducing class noise before retraining.

## Applied Merges
- `练习零件图` → `机械制图`
- `轴类` → `机械制图`
- `模板` → `机械制图`
- `盖` → `机械制图`
- `视图` → `机械制图`
- `三视图` → `机械制图`
- `紧固件` → `机械制图`
- `板类` → `机械制图`
- `示意图` → `机械制图`
- `轴的标注` → `机械制图`

## Results
- Manifest rows: 223
- Unique labels: 84 → 74
- Top label: `机械制图` (118)

## Outputs
- `reports/MECH_4000_DWG_LABEL_MANIFEST_MERGED2_20260120.csv`
- `data/knowledge/geometry_rules.json`
