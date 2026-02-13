# DEV_GRAPH2D_LOCAL_RETRAIN_TRAINING_DRAWINGS_COARSE_20260213

## Goal

Use the new local Graph2D pipeline to retrain a coarse-bucket Graph2D model on the local training DXFs, and capture reproducible validation + diagnosis metrics without manual DXF inspection.

## Dataset

- DXF dir: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`
- Samples: 110 DXFs
- Weak labels: extracted from filenames, then normalized into 11 coarse buckets and cleaned with `--clean-min-count 2`.

Coarse bucket distribution (post-normalize):

- `传动件` 21
- `设备` 19
- `罐体` 18
- `轴承件` 11
- `法兰` 11
- `罩盖件` 8
- `过滤组件` 8
- `开孔件` 4
- `支撑件` 4
- `弹簧` 3
- `紧固件` 3

## Command

```bash
.venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --normalize-labels \
  --clean-min-count 2 \
  --model edge_sage \
  --loss cross_entropy \
  --class-weighting inverse \
  --sampler balanced \
  --epochs 15 \
  --diagnose-max-files 80
```

## Outputs

- work dir: `/tmp/graph2d_pipeline_local_20260213_192150`
- checkpoint: `/tmp/graph2d_pipeline_local_20260213_192150/graph2d_trained.pth`
- eval:
  - metrics: `/tmp/graph2d_pipeline_local_20260213_192150/eval_metrics.csv`
  - errors: `/tmp/graph2d_pipeline_local_20260213_192150/eval_errors.csv`
- diagnose:
  - summary: `/tmp/graph2d_pipeline_local_20260213_192150/diagnose/summary.json`

## Results

### Validation (eval_2d_graph.py)

Overall on validation split (`__overall__` row):

- `acc`: **0.450** (9/20)
- `top2_accuracy`: **0.700**
- `macro_f1`: **0.391**
- `weighted_f1`: **0.368**

### Diagnosis (diagnose_graph2d_on_dxf_dir.py)

Scored against manifest truth (coarse buckets):

- sampled files: 80
- `accuracy`: **0.375**
- confidence p50/p90: **0.286 / 0.579**

## Notes / Interpretation

- This is a **coarse-label** model. Metrics are relative to filename-derived weak labels after normalization/cleaning.
- It confirms the local pipeline is functional end-to-end and produces non-trivial performance in the coarse label space.

