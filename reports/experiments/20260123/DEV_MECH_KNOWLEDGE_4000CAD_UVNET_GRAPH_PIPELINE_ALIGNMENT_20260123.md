# DEV_MECH_KNOWLEDGE_4000CAD_UVNET_GRAPH_PIPELINE_ALIGNMENT_20260123

## Summary
- Threaded `edge_attr` through graph collation and training/inference paths.
- Aligned model defaults to infer node input dimensions from schema when available.
- Added an MPS-safe dense adjacency fallback for the pure-torch GCN path.
- Ensured PyG samples attach scalar labels and training scripts persist schema in checkpoints.

## Changes
- `src/ml/train/trainer.py` (edge_attr collation, PyG/DataLoader selection)
- `src/ml/train/model.py` (edge_attr-aware forward signature, schema-based defaults)
- `src/ml/vision_3d.py` (schema-driven defaults, edge_attr passthrough)
- `src/ml/train/dataset.py` (scalar PyG labels)
- `scripts/train_uvnet_graph.py` (persist schema in config)
- `scripts/train_uvnet_graph_dryrun.py` (edge_attr passthrough)
- `tests/test_uvnet_graph_flow.py` (edge_attr coverage)

## Notes
- Edge attributes are now available end-to-end but are not yet consumed by the GCN layers.
