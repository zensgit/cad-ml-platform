# DEV_MECH_KNOWLEDGE_4000CAD_UVNET_GRAPH_EDGE_ATTR_WEIGHTING_20260123

## Summary
- Added edge attribute weighting to the UV-Net GCN path.
- Edge attributes now map to scalar edge weights via a learned linear gate.

## Changes
- `src/ml/train/model.py` (edge_attr weighting in PyG/pure-torch paths)
- `src/ml/train/trainer.py` (edge_attr dimension validation)
- `src/ml/vision_3d.py` (edge_input_dim from checkpoint config)

## Notes
- Edge weights are applied uniformly across GCN layers using the same edge_attr-derived weights.
