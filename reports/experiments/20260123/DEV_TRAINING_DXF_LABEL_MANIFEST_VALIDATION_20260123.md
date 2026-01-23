# DEV_TRAINING_DXF_LABEL_MANIFEST_VALIDATION_20260123

## Checks
- Ran the DXF label manifest builder against the training DXF directory.
- Summarized label coverage for the generated CSV.

## Runtime Output
- Command:
  - `python3 scripts/build_dxf_label_manifest.py --input-dir "$DXF_DIR" --output-csv reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_20260123.csv`
- Result:
  - `rows=110`
  - `labels=47`
  - `missing_labels=0`
  - Top labels: 罐体部分(5), 上封头组件(5), 过滤托架(4), 蜗轮蜗杆传动出料机构(3), 短轴承座(盖)(3)

## Notes
- The manifest is ready for use with `scripts/train_2d_graph.py`.
