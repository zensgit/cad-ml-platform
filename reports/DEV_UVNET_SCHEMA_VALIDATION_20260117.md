# DEV_UVNET_SCHEMA_VALIDATION_20260117

## Summary
Recorded graph schema metadata in UV-Net checkpoints and validated inference
inputs when schema information is provided.

## Design
- Doc: `docs/UVNET_SCHEMA_VALIDATION.md`

## Steps
- Added `node_schema`/`edge_schema` to `UVNetGraphModel` config.
- `UVNetEncoder.encode()` now checks input schema against checkpoint schema.
- Added unit coverage for schema mismatch handling.
- Ran: `source .venv-graph/bin/activate && pytest tests/unit/test_uvnet_encoder_dimension_guard.py -v`.

## Results
- Tests passed (dimension mismatch and schema mismatch guards).

## Notes
- Schema validation is best-effort and only enforced when both model and input
  provide schema metadata.
