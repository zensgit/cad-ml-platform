# Hybrid Real Data Validation 2026-03-06

## Scope

This validation compares branch-level DXF classification behavior on a real
labeled dataset using the full local `/api/v1/analyze/` pipeline.

The goal is not just to score Graph2D, but to measure how much value is added
by:

- `graph2d`
- `filename`
- `titleblock`
- `hybrid`
- final API `part_type`

## Data Used

- Manifest: `reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_CLEANED_20260123.csv`
- Dataset size: `110`
- Coarse labels: `8`
- Graph2D checkpoint: local real-data checkpoint generated on `2026-03-06`

## Command

```bash
python3 scripts/eval_hybrid_dxf_manifest.py \
  --dxf-dir /path/to/local/训练图纸_dxf_oda_20260123 \
  --manifest reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_CLEANED_20260123.csv \
  --graph2d-model-path models/graph2d_real_validation_oda110_20260306.pth \
  --output-dir reports/experiments/20260306/hybrid_real_validation_oda110 \
  --max-files 110
```

## Local-Only Artifacts

The following outputs were generated locally and intentionally not committed:

- `reports/experiments/20260306/hybrid_real_validation_oda110/results.csv`
- `reports/experiments/20260306/hybrid_real_validation_oda110/summary.json`

## Result Summary

Elapsed time:

- `78.876s`

Branch accuracy against manifest labels:

1. `filename_label`: `0.8727`
2. `titleblock_label`: `0.8727`
3. `hybrid_label`: `0.8727`
4. `fine_part_type`: `0.8727`
5. `final_part_type`: `0.5545`
6. `graph2d_label`: `0.1182`

Confidence summary:

- `graph2d_label`
  - `p50 = 0.141236`
  - `p90 = 0.145677`
  - `low_conf_rate = 1.0`
- `filename_label`
  - `p50 = 0.95`
  - `p90 = 0.95`
  - `low_conf_rate = 0.0`
- `titleblock_label`
  - `p50 = 0.85`
  - `p90 = 0.85`
  - `low_conf_rate = 0.0`
- `hybrid_label`
  - `p50 = 0.95`
  - `p90 = 0.95`
  - `low_conf_rate = 0.0`
- `final_part_type`
  - `p50 = 0.95`
  - `p90 = 0.95`
  - `low_conf_rate = 0.0`

## Findings

1. `Graph2D` remains weak on this real DXF set.
   - Accuracy stays at `0.1182`
   - `low_conf_rate = 1.0`
   - This matches the standalone Graph2D diagnosis from the same dataset

2. `filename` and `titleblock` are the dominant useful signals.
   - Both independently reach `0.8727`
   - On this dataset, the hybrid branch is mostly carried by these signals

3. `hybrid_label` is strong, but `final_part_type` is materially worse.
   - `hybrid_label = 0.8727`
   - `final_part_type = 0.5545`
   - This gap is caused by output-shape semantics, not branch quality

4. The current API final field is not a reliable coarse-family output for DXF.
   - Many `part_type` values remain fine-grained or rule-shaped labels such as:
     - `盖`
     - `轴承`
     - `支腿`
     - `挡板`
     - `旋转组件`
   - These do not align cleanly with the coarse manifest buckets used by Graph2D
     training and validation

## Representative Examples

Examples where `hybrid/fine` is semantically right but `part_type` is not aligned
to the coarse bucket:

1. `支承座`
   - true coarse label: `轴承件`
   - `hybrid_label`: `支承座`
   - `part_type`: `盖`

2. `汽水分离器`
   - true coarse label: `设备`
   - `hybrid_label`: `汽水分离器`
   - `part_type`: `支腿`

3. `搅拌轴组件`
   - true coarse label: `传动件`
   - `hybrid_label`: `搅拌轴组件`
   - `part_type`: `旋转组件`

## Interpretation

This real-data run changes the engineering conclusion:

- If the task is coarse DXF family classification, `Graph2D` is not the main
  value carrier yet.
- The strongest current production path is `filename + titleblock + hybrid`.
- The next bottleneck is output normalization, not branch confidence.

In practical terms:

1. Keep `Graph2D` as a weak signal and review/rejection input.
2. Trust `hybrid/fine` far more than standalone `graph2d`.
3. Add an explicit coarse-normalized output field for DXF API results, rather
   than overloading `part_type` to serve both fine and coarse semantics.

## Recommended Next Step

Add additive coarse output fields to DXF analysis responses, for example:

- `coarse_part_type`
- `coarse_hybrid_label`
- `coarse_filename_label`
- `coarse_titleblock_label`
- `coarse_graph2d_label`

This would let downstream systems consume a stable coarse taxonomy without
discarding the existing fine-grained hybrid outputs.
