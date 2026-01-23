# DEV_MECH_KNOWLEDGE_4000CAD_UVNET_GRAPH_PYG_EDGE_ATTR_UPGRADE_20260123

## Summary
- Added optional PyG edge-attribute message passing (GINEConv) when torch_geometric is available.
- Preserved weighted-GCN fallback for pure torch environments.
- Validated the PyG path in `.venv-graph` with a synthetic training run.

## Changes
- `src/ml/train/model.py` (GINEConv path + edge backend metadata)

## Notes
- In environments without PyG, the model continues to use edge_attr-weighted GCN.
