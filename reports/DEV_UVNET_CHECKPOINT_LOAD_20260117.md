# DEV_UVNET_CHECKPOINT_LOAD_20260117

## Summary
Validated that the UV-Net encoder can load the smoke-test checkpoint and produce
an embedding for graph inputs.

## Steps
- Ran: `source .venv-graph/bin/activate && python3 - <<'PY' ... PY` to load
  `models/smoke_test_model.pth` with `UVNetEncoder` and generate an embedding.

## Results
- Initial run with a 12-dim node feature tensor failed due to a shape mismatch
  (expected, since the checkpoint was trained with 15-dim inputs).
- Re-run with 15-dim node features succeeded and returned a 1024-dim embedding.

## Notes
- Node feature dimensionality must match the checkpoint config (`node_input_dim`).
- On inference errors, the encoder returns a zero vector for safety.
