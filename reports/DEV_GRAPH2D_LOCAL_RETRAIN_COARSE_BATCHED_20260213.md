# DEV_GRAPH2D_LOCAL_RETRAIN_COARSE_BATCHED_20260213

## Goal

Re-run the coarse-bucket Graph2D local pipeline after enabling batched training/evaluation (no PyG dependency) to confirm:

- End-to-end pipeline still works.
- Iteration speed improves substantially vs the per-sample training loop.
- Baseline quality metrics remain comparable (or improve) with the faster loop.

## Command

```bash
/usr/bin/time -p .venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --normalize-labels \
  --clean-min-count 2 \
  --model edge_sage \
  --loss cross_entropy \
  --class-weighting inverse \
  --sampler balanced \
  --epochs 15 \
  --diagnose-max-files 80 \
  --graph-cache both \
  --empty-edge-fallback knn \
  --empty-edge-knn-k 8
```

Artifacts:

- Work dir: `/tmp/graph2d_pipeline_local_20260213_233141`
- Checkpoint: `/tmp/graph2d_pipeline_local_20260213_233141/graph2d_trained.pth`
- Pipeline summary: `/tmp/graph2d_pipeline_local_20260213_233141/pipeline_summary.json`

## Results

### Train (val accuracy during training)

- Best observed `val_acc`: `0.259` (epoch 13)
- Final epoch `val_acc`: `0.222`

### Eval (validation split)

From `scripts/eval_2d_graph.py`:

- `acc=0.350`
- `top2=0.600`
- `macro_f1=0.311`
- `weighted_f1=0.285`

### Diagnose (manifest-truth scoring, sampled 80)

From `scripts/diagnose_graph2d_on_dxf_dir.py --manifest-csv ...`:

- Accuracy vs manifest truth: `0.275` (22 / 80)
- Confidence distribution: p50/p90 `0.279 / 0.524`
- Top predicted labels:
  - `紧固件`: 24
  - `罐体`: 17
  - `开孔件`: 15
  - `传动件`: 8
  - `过滤组件`: 7
  - `支撑件`: 6

Key observed confusions:

- `法兰 -> 紧固件` (8)
- `设备 -> 罐体` (5)
- `轴承件 -> 紧固件` (4)
- `传动件 -> 紧固件` (4)

## Timing

`time -p`:

- `real 43.07`
- `user 35.17`
- `sys 8.71`

## Notes

- The model no longer collapses to a single class in this run, but it still over-predicts `紧固件` relative to its true frequency.
- Follow-up action (next experiment): try `scripts/train_2d_graph.py --downweight-label 紧固件 --downweight-factor 0.3` to reduce the most common false-positive class and re-check the diagnosis confusions.

