# DEV_UVNET_CHECKPOINT_CONFIG_20260117

## Summary
Aligned UV-Net checkpoint configuration with inference loading to prevent
architecture mismatches and validated the graph flow tests.

## Design
- Doc: `docs/UVNET_MODEL_CHECKPOINT_CONFIG.md`

## Steps
- Added `num_classes` and `dropout_rate` to model config metadata.
- Updated `UVNetEncoder` to load `hidden_dim`, `num_classes`, and `dropout_rate` from
  checkpoint config.
- Adjusted the UV-Net graph flow test to avoid BatchNorm failures on single-graph
  inference.
- Ran: `source .venv-graph/bin/activate && pytest tests/test_uvnet_graph_flow.py -v`.

## Results
- `tests/test_uvnet_graph_flow.py` passed (3 tests).

## Notes
- New config fields are backwards compatible; defaults apply if absent in older
  checkpoints.
