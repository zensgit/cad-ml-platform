# DEV_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_20260123

## Summary
- Normalized ODA DXF labels into 11 canonical buckets using explicit mapping rules.

## Command
- `.venv-graph/bin/python scripts/normalize_dxf_label_manifest.py --input-csv "reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_20260123.csv" --output-csv "reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_20260123.csv" --strict`

## Output
- `reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_20260123.csv`

## Label Counts
- 传动件=21, 设备=19, 罐体=18, 轴承件=11, 法兰=11
- 罩盖件=8, 过滤组件=8, 开孔件=4, 支撑件=4, 弹簧=3, 紧固件=3
