# DEV_GRAPH2D_COARSE_SYNTH_V2_TRAIN_EVAL_20260208

## Goal
Move Graph2D toward a **coarse taxonomy** objective (5 classes) so it can serve as a
safe fallback/aux signal, while Hybrid keeps producing fine-grained labels from
filename/titleblock/process.

Also ensure Graph2D coarse labels do not trigger misleading `soft_override_suggestion`
recommendations.

## Changes
- `config/hybrid_classifier.yaml`
  - Relaxed `graph2d.exclude_labels` back to `other` so coarse family labels are not
    filtered by default.

- `src/api/v1/analyze.py`
  - Default `GRAPH2D_MIN_CONF` to Hybrid config (`config/hybrid_classifier.yaml`) when the
    env var is absent (prevents low-confidence Graph2D from being treated as fusable).
  - Added `graph2d_prediction.is_coarse_label` (default coarse labels:
    `传动件/壳体类/轴类/连接件/其他`; override via `GRAPH2D_COARSE_LABELS`).
  - Prevented `soft_override_suggestion` from becoming eligible for coarse Graph2D labels
    (reason: `graph2d_coarse_label`).

## Training / Evaluation (Synthetic v2)
Dataset:
- `data/synthetic_v2` (parent-dir labels: `传动件/壳体类/轴类/连接件/其他`)

### 1) Build manifest
```bash
python3 scripts/build_dxf_label_manifest.py \
  --input-dir data/synthetic_v2 \
  --recursive \
  --label-mode parent_dir \
  --no-standardize \
  --output-csv reports/experiments/20260208/DXF_LABEL_MANIFEST_SYNTH_V2_COARSE_20260208.csv
```

Result: `4000` rows written.

### 2) Train checkpoint (GCN, 5-class)
```bash
mkdir -p /tmp/pycache /tmp/xdg-cache
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPYCACHEPREFIX=/tmp/pycache \
XDG_CACHE_HOME=/tmp/xdg-cache \
python3 scripts/train_2d_graph.py \
  --manifest reports/experiments/20260208/DXF_LABEL_MANIFEST_SYNTH_V2_COARSE_20260208.csv \
  --dxf-dir data/synthetic_v2 \
  --model gcn \
  --epochs 8 \
  --batch-size 8 \
  --max-samples 1000 \
  --max-samples-strategy stratified \
  --save-best \
  --output models/graph2d_coarse_synth_v2_20260208.pth
```

Result (best): `val_acc=0.989`

Note: the checkpoint is generated locally at `models/graph2d_coarse_synth_v2_20260208.pth`
(`28K`). It is ignored by default git rules (`models/*.pth`).

### 3) Eval checkpoint (per-class metrics)
```bash
mkdir -p /tmp/pycache /tmp/xdg-cache
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPYCACHEPREFIX=/tmp/pycache \
XDG_CACHE_HOME=/tmp/xdg-cache \
python3 scripts/eval_2d_graph.py \
  --manifest reports/experiments/20260208/DXF_LABEL_MANIFEST_SYNTH_V2_COARSE_20260208.csv \
  --dxf-dir data/synthetic_v2 \
  --checkpoint models/graph2d_coarse_synth_v2_20260208.pth \
  --val-split 0.2 \
  --split-strategy stratified \
  --output-metrics reports/experiments/20260208/SYNTH_V2_GRAPH2D_COARSE_VAL_METRICS_20260208.csv \
  --output-errors reports/experiments/20260208/SYNTH_V2_GRAPH2D_COARSE_VAL_ERRORS_20260208.csv
```

Result summary:
- `acc=0.990`
- `top2=0.997`
- `macro_f1=0.988`
- `weighted_f1=0.989`

Artifacts:
- `reports/experiments/20260208/DXF_LABEL_MANIFEST_SYNTH_V2_COARSE_20260208.csv`
- `reports/experiments/20260208/SYNTH_V2_GRAPH2D_COARSE_VAL_METRICS_20260208.csv`
- `reports/experiments/20260208/SYNTH_V2_GRAPH2D_COARSE_VAL_ERRORS_20260208.csv`

## How To Use This Checkpoint
Enable Graph2D and point it at the coarse checkpoint:
```bash
GRAPH2D_ENABLED=true \
GRAPH2D_MODEL_PATH=models/graph2d_coarse_synth_v2_20260208.pth \
uvicorn src.main:app --reload
```

