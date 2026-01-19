# DEV_UVNET_ENCODER_DIMENSION_GUARD_20260117

## Summary
Added a node-dimension guard to the UV-Net encoder and validated behavior with a
unit test.

## Design
- Doc: `docs/UVNET_ENCODER_DIMENSION_CHECK.md`

## Steps
- Updated `src/ml/vision_3d.py` to validate node feature shape and dimension.
- Added `tests/unit/test_uvnet_encoder_dimension_guard.py`.
- Ran: `source .venv-graph/bin/activate && pytest tests/unit/test_uvnet_encoder_dimension_guard.py -v`.

## Results
- Test passed.

## Notes
- When dimensions mismatch, the encoder now returns a zero embedding of the
  expected size.
