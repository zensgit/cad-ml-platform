#!/usr/bin/env markdown
# 3D Training Pipeline

This document describes the available training scripts for 3D models and their
expected data layouts.

## UV-Net Scaffold Training

Script: `src/ml/train/trainer.py`

```
python3 src/ml/train/trainer.py --data-dir data/abc_subset --epochs 5 --dry-run
```

Expected input: STEP files under `data/abc_subset/`.

## Hybrid 3D Model Training

Script: `scripts/train_hybrid_3d_model.py`

```
python3 scripts/train_hybrid_3d_model.py --data-dir data/training_3d --epochs 10
```

Expected data layout:

```
data/training_3d/
  features/ {id}.npy   # 160-dim v7 features
  points/   {id}.npy   # Nx3 point clouds
  labels.csv           # id,label_idx
```

## Notes

- These scripts require PyTorch.
- For CI or smoke checks, use `--dry-run` where available.
