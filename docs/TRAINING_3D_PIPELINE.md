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

## UV-Net Smoke Test

Script: `scripts/train_smoke_test.py`

```
source .venv-graph/bin/activate
python3 scripts/train_smoke_test.py
```

Expected behavior:

- Generates a synthetic graph dataset for quick end-to-end checks.
- Trains for 5 epochs on the detected device.
- Writes a checkpoint to `models/smoke_test_model.pth`.

## UV-Net Graph Training Loop

Script: `scripts/train_uvnet_graph.py`

```
source .venv-graph/bin/activate
python3 scripts/train_uvnet_graph.py --data-dir data/abc_subset --epochs 10
```

Expected behavior:

- Builds graph samples from STEP files and trains the UV-Net graph model end-to-end.
- Uses `label_strategy=surface_bucket` by default to create pseudo-labels from surface counts.
- Writes a checkpoint to `models/uvnet_graph_latest.pth` unless overridden via `--output`.
- Use `--synthetic` to run the loop without STEP parsing (no pythonocc required).

## UV-Net Checkpoint Inspector

Script: `scripts/uvnet_checkpoint_inspect.py`

```
source .venv-graph/bin/activate
python3 scripts/uvnet_checkpoint_inspect.py --path models/smoke_test_model.pth
```

Expected behavior:

- Prints checkpoint config fields.
- Runs a minimal forward pass and reports output shapes.

## UV-Net Graph Dry-Run

Script: `scripts/train_uvnet_graph_dryrun.py`

```
source .venv-graph/bin/activate
python3 scripts/train_uvnet_graph_dryrun.py --data-dir data/abc_subset
```

Expected behavior:

- Skips with a friendly message if `pythonocc-core` is unavailable.
- Loads graph data from STEP files and runs a single forward pass.

## Notes

- These scripts require PyTorch.
- For CI or smoke checks, use `--dry-run` where available.
