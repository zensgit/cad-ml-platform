# DEV_UVNET_TRAINER_INPUT_GUARD_20260117

## Summary
Added a training-time guard that rejects mismatched node feature dimensions.

## Design
- Doc: `docs/UVNET_TRAINER_INPUT_GUARD.md`

## Steps
- Added node feature dimension validation in `UVNetTrainer`.
- Added `tests/unit/test_uvnet_trainer_input_guard.py`.
- Ran: `source .venv-graph/bin/activate && pytest tests/unit/test_uvnet_trainer_input_guard.py -v`.

## Results
- Test passed.

## Notes
- The guard fails fast with a clear error if dataset features do not align with
  model expectations.
