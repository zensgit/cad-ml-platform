# DEV_MECH_KNOWLEDGE_4000CAD_UVNET_GRAPH_PYG_EDGE_ATTR_UPGRADE_VALIDATION_20260123

## Checks
- Installed torch_geometric in `.venv-graph` and confirmed GINEConv is available.
- Verified model config reports `edge_backend=gine` when PyG is present.
- Ran a synthetic training loop with PyG enabled.

## Runtime Output
- Command:
  - `.venv-graph/bin/pip install torch-geometric`
- Result:
  - `torch-geometric-2.7.0` installed in `.venv-graph`
- Command:
  - `.venv-graph/bin/python - <<'PY'\nfrom src.core.geometry.engine import BREP_GRAPH_NODE_FEATURES, BREP_GRAPH_EDGE_FEATURES\nfrom src.ml.train.model import UVNetGraphModel\n\nmodel = UVNetGraphModel(\n    node_input_dim=len(BREP_GRAPH_NODE_FEATURES),\n    edge_input_dim=len(BREP_GRAPH_EDGE_FEATURES),\n    node_schema=BREP_GRAPH_NODE_FEATURES,\n    edge_schema=BREP_GRAPH_EDGE_FEATURES,\n)\nprint(model.get_config())\nPY`
- Result:
  - `{'edge_backend': 'gine', 'backend': 'pyg', ...}`
- Command:
  - `.venv-graph/bin/python scripts/train_uvnet_graph.py --synthetic --synthetic-samples 12 --epochs 1 --batch-size 4 --output /tmp/uvnet_graph_pyg_gine_smoke.pth`
- Result:
  - `Epoch 1/1 loss=1.5068 acc=0.5000 val_loss=1.5670 val_acc=0.5000 time=1.75s`

## Notes
- STEP parsing is still unavailable in this environment due to missing `pythonocc-core`.
