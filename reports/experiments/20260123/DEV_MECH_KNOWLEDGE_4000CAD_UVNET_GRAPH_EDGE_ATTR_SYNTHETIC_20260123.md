# DEV_MECH_KNOWLEDGE_4000CAD_UVNET_GRAPH_EDGE_ATTR_SYNTHETIC_20260123

## Summary
- Added edge attribute generation to the synthetic UV-Net graph dataset to exercise edge_attr plumbing.
- Expanded UV-Net graph flow test to pass edge_attr into the model forward call.

## Changes
- `scripts/train_uvnet_graph.py` (synthetic dataset edge_attr)
- `tests/test_uvnet_graph_flow.py` (edge_attr forward coverage)

## Notes
- Edge attributes are randomly generated in synthetic mode to validate pipeline wiring.
