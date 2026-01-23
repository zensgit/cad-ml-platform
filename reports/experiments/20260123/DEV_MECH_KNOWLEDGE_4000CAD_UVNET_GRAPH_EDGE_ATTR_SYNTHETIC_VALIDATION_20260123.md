# DEV_MECH_KNOWLEDGE_4000CAD_UVNET_GRAPH_EDGE_ATTR_SYNTHETIC_VALIDATION_20260123

## Checks
- Ran a synthetic UV-Net graph training loop with edge attributes enabled.

## Runtime Output
- Command:
  - `python3 scripts/train_uvnet_graph.py --synthetic --synthetic-samples 12 --epochs 1 --batch-size 4 --output /tmp/uvnet_graph_edge_attr_smoke.pth`
- Result:
  - `Epoch 1/1 loss=1.7119 acc=0.1250 val_loss=1.5891 val_acc=0.5000 time=0.50s`

## Notes
- STEP parsing remains unavailable in this environment due to missing `pythonocc-core`.
