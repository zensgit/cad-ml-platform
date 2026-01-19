# DEV_UVNET_TRAINING_SCHEMA_ALIGN_20260117

## Summary
Aligned UV-Net training scripts with B-Rep graph schemas and validated the
checkpoint now records schema metadata.

## Design
- Doc: `docs/UVNET_SCHEMA_VALIDATION.md`

## Steps
- Updated `scripts/train_smoke_test.py` and `scripts/train_uvnet_graph_dryrun.py`
  to pass `node_schema`/`edge_schema` into `UVNetGraphModel`.
- Ran: `source .venv-graph/bin/activate && python3 scripts/train_smoke_test.py`.
- Ran: `source .venv-graph/bin/activate && python3 scripts/uvnet_checkpoint_inspect.py --path models/smoke_test_model.pth`.

## Results
- Smoke training completed and checkpoint saved.
- Checkpoint config now includes node/edge schema tuples.

## Notes
- `pythonocc-core` is still unavailable locally; schema values are sourced from
  `BREP_GRAPH_*_FEATURES`.
