#!/usr/bin/env markdown
# UV-Net Checkpoint Config

## Goal
Ensure UV-Net checkpoints carry enough architecture metadata to reload the model
for inference without shape mismatches.

## Config Fields
Saved in `UVNetGraphModel.get_config()` and stored in the checkpoint:

- `node_input_dim`: Face feature dimension.
- `hidden_dim`: GCN hidden width.
- `embedding_dim`: Output embedding size.
- `num_classes`: Classifier output size.
- `dropout_rate`: MLP dropout rate.
- `node_schema`: Optional node feature schema tuple.
- `edge_schema`: Optional edge feature schema tuple.
- `backend`: `pyg` or `pure_torch` (informational).

## Inference Load Behavior
`UVNetEncoder._load_model()` now uses these fields to reconstruct the exact
architecture. Missing fields fall back to the existing defaults to preserve
backward compatibility with older checkpoints.

## Compatibility Notes
- If a checkpoint was saved before these fields existed, the defaults apply and
  the load still succeeds.
- If a checkpoint was saved with non-default dimensions, load will now match
  those shapes, preventing `state_dict` size mismatches.
