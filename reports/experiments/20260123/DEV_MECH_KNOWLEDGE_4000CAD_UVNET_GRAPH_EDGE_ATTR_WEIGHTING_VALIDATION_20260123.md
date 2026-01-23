# DEV_MECH_KNOWLEDGE_4000CAD_UVNET_GRAPH_EDGE_ATTR_WEIGHTING_VALIDATION_20260123

## Checks
- Ran a synthetic UV-Net training loop to verify edge_attr weighting executes end-to-end.

## Runtime Output
- Command:
  - `python3 scripts/train_uvnet_graph.py --synthetic --synthetic-samples 12 --epochs 1 --batch-size 4 --output /tmp/uvnet_graph_edge_attr_weighted_smoke.pth`
- Result:
  - `Epoch 1/1 loss=1.7514 acc=0.3750 val_loss=1.5955 val_acc=0.5000 time=1.37s`

## Notes
- STEP parsing is still unavailable in this environment due to missing `pythonocc-core`.
