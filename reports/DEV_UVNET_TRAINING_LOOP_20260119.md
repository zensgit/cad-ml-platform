# DEV_UVNET_TRAINING_LOOP_20260119

## Summary
- Added a UV-Net graph training loop script that runs an end-to-end train/val cycle on STEP graph data.
- Introduced a surface-count bucket label strategy for pseudo-supervised L4 training.
- Updated the 3D training pipeline documentation.

## Changes
- `src/ml/train/dataset.py`: added `label_strategy` support with surface-count buckets.
- `scripts/train_uvnet_graph.py`: added the L4 training loop with train/val split and checkpoint output.
- `docs/TRAINING_3D_PIPELINE.md`: documented the new training loop and synthetic mode.
  - Ensured training uses `drop_last=True` to avoid batchnorm failures on the final short batch.

## Validation
- Ran with the synthetic dataset in the existing `.venv-graph` torch environment:
  - `./.venv-graph/bin/python scripts/train_uvnet_graph.py --synthetic --epochs 1 --batch-size 4 --synthetic-samples 16`
- Output:
  - `Epoch 1/1 loss=1.7431 acc=0.0833 val_loss=1.5604 val_acc=0.6667 time=0.89s`
