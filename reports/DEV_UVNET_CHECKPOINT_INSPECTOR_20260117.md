# DEV_UVNET_CHECKPOINT_INSPECTOR_20260117

## Summary
Added a UV-Net checkpoint inspector and verified it against the smoke-test
checkpoint.

## Design
- Doc: `docs/UVNET_CHECKPOINT_INSPECTOR.md`

## Steps
- Added `scripts/uvnet_checkpoint_inspect.py`.
- Ran: `source .venv-graph/bin/activate && python3 scripts/uvnet_checkpoint_inspect.py --path models/smoke_test_model.pth`.

## Results
- Script printed checkpoint config and successfully ran a minimal forward pass.

## Notes
- Uses the pure torch backend if PyG is not installed.
