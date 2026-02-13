# DEV_GRAPH2D_DOWNWEIGHT_LABEL_EXPERIMENT_20260213

## Goal

Test whether downweighting the most common false-positive label (`紧固件`) improves coarse-bucket Graph2D performance and reduces confusion, using the same manifest + hyperparameters as the batched coarse retrain baseline.

## Baseline (for comparison)

Baseline report: `reports/DEV_GRAPH2D_LOCAL_RETRAIN_COARSE_BATCHED_20260213.md`

Key baseline results:

- Eval (val split): `acc=0.350`, `top2=0.600`, `macro_f1=0.311`, `weighted_f1=0.285`
- Diagnose (manifest truth, sampled 80): `accuracy=0.275`
- Top predicted labels (diagnose): `紧固件=24`, `罐体=17`, `开孔件=15`

## Experiment Setup

Inputs reused from baseline pipeline run:

- Manifest: `/tmp/graph2d_pipeline_local_20260213_233141/manifest.cleaned.csv`
- DXF dir: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`

Environment:

- `DXF_MANIFEST_DATASET_CACHE=both`
- `DXF_EMPTY_EDGE_FALLBACK=knn` (`DXF_EMPTY_EDGE_K=8`)
- Sampling: `DXF_MAX_NODES=200`, `DXF_SAMPLING_STRATEGY=importance`, `DXF_SAMPLING_SEED=42`, `DXF_TEXT_PRIORITY_RATIO=0.3`

## Commands

```bash
WORK_DIR=/tmp/graph2d_downweight_20260213_233520
MANIFEST=/tmp/graph2d_pipeline_local_20260213_233141/manifest.cleaned.csv
DXF_DIR="/Users/huazhou/Downloads/训练图纸/训练图纸_dxf"

DXF_MANIFEST_DATASET_CACHE=both \
DXF_MANIFEST_DATASET_CACHE_DIR="$WORK_DIR/graph_cache" \
DXF_EMPTY_EDGE_FALLBACK=knn \
DXF_EMPTY_EDGE_K=8 \
DXF_MAX_NODES=200 \
DXF_SAMPLING_STRATEGY=importance \
DXF_SAMPLING_SEED=42 \
DXF_TEXT_PRIORITY_RATIO=0.3 \
.venv/bin/python scripts/train_2d_graph.py \
  --manifest "$MANIFEST" \
  --dxf-dir "$DXF_DIR" \
  --epochs 15 \
  --batch-size 4 \
  --hidden-dim 64 \
  --lr 0.001 \
  --model edge_sage \
  --loss cross_entropy \
  --class-weighting inverse \
  --sampler balanced \
  --seed 42 \
  --device cpu \
  --output "$WORK_DIR/graph2d_trained.pth" \
  --dxf-max-nodes 200 \
  --dxf-sampling-strategy importance \
  --dxf-sampling-seed 42 \
  --dxf-text-priority-ratio 0.3 \
  --downweight-label "紧固件" \
  --downweight-factor 0.3
```

Then:

- `scripts/eval_2d_graph.py` on the same manifest
- `scripts/diagnose_graph2d_on_dxf_dir.py --manifest-csv` sampled 80

## Results

### Timing

- Train: `real 21.30s`
- Eval: `real 1.33s`
- Diagnose: `real 12.56s`

### Eval (val split)

- `acc=0.300`
- `top2=0.500`
- `macro_f1=0.300`
- `weighted_f1=0.251`

### Diagnose (manifest truth, sampled 80)

- Accuracy vs manifest truth: `0.225`
- Confidence p50/p90: `0.271 / 0.486`
- Top predicted labels:
  - `开孔件`: 20
  - `紧固件`: 17
  - `罐体`: 14
  - `过滤组件`: 11

Notable changes vs baseline:

- `紧固件` predictions decreased (`24 -> 17`) but overall accuracy dropped (`0.275 -> 0.225`).
- The biggest weak classes remained weak (`设备/轴承件/法兰` still near `0.0` per-class accuracy in the sampled diagnose set).

## Conclusion

Downweighting `紧固件` at factor `0.3` reduced that label's prediction frequency, but did not improve overall accuracy and slightly degraded both eval and manifest-truth diagnosis accuracy.

Recommended next iteration (if we continue improving geometry-only Graph2D):

- Try alternative imbalance strategies (`--loss focal` or `--loss logit_adjusted`) and/or a less aggressive downweight factor.
- Consider distillation or multi-signal training where we explicitly evaluate a “no-filename” scenario (mask filenames during teacher signal generation) to make the student useful when filenames are absent.

