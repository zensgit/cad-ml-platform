#!/usr/bin/env markdown
# UV-Net Trainer Input Guard

## Goal
Fail fast during training if graph node feature dimensionality does not match
model expectations.

## Behavior
- `UVNetTrainer` validates node feature tensors are 2D.
- The trainer raises a `ValueError` if `x.shape[1]` does not match
  `model.node_input_dim`.

## Notes
This guard prevents silent training failures due to misconfigured datasets.
