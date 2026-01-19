# DEV_UVNET_SMOKE_TEST_NODE_DIM_20260117

## Summary
Aligned the UV-Net smoke test node dimension with the B-Rep graph schema and
re-ran the smoke training to confirm consistency.

## Design
- Doc: `docs/UVNET_SMOKE_TEST_NODE_DIM.md`

## Steps
- Updated `scripts/train_smoke_test.py` to derive `DEFAULT_NODE_DIM` from
  `BREP_GRAPH_NODE_FEATURES`.
- Ran: `source .venv-graph/bin/activate && python3 scripts/train_smoke_test.py`.

## Results
- Training completed on MPS for 5 epochs.
- Checkpoint written to `models/smoke_test_model.pth`.

## Notes
- `pythonocc-core` is still absent locally; warnings are expected.
