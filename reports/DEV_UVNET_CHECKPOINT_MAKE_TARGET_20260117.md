# DEV_UVNET_CHECKPOINT_MAKE_TARGET_20260117

## Summary
Added a Make target for the UV-Net checkpoint inspector.

## Design
- Doc: `docs/UVNET_CHECKPOINT_INSPECTOR.md`

## Steps
- Added `uvnet-checkpoint-inspect` to `Makefile`.
- Ran: `make uvnet-checkpoint-inspect UVNET_CHECKPOINT=models/smoke_test_model.pth PYTHON=.venv-graph/bin/python`.

## Results
- Inspector printed checkpoint config and output shapes.
