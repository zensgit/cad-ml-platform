# DEV_UVNET_SMOKE_TRAINING_20260117

## Summary
Ran the UV-Net smoke training script to validate end-to-end graph training and
checkpoint generation.

## Design
- Doc: `docs/TRAINING_3D_PIPELINE.md`

## Steps
- Ran: `source .venv-graph/bin/activate && python3 scripts/train_smoke_test.py`.

## Results
- Training completed on MPS for 5 epochs.
- Checkpoint written to `models/smoke_test_model.pth`.

## Notes
- The dataset is synthetic and intended for pipeline validation only.
