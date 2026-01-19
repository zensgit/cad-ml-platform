# DEV_UVNET_GRAPH_DRYRUN_20260117

## Summary
Added and exercised a dry-run script that loads STEP-based graph data and runs a
single UV-Net forward pass.

## Design
- Doc: `docs/UVNET_GRAPH_DRYRUN.md`

## Steps
- Added `scripts/train_uvnet_graph_dryrun.py`.
- Ran: `source .venv-graph/bin/activate && python3 scripts/train_uvnet_graph_dryrun.py --data-dir data/abc_subset`.

## Results
- Dry-run skipped locally because `pythonocc-core` is not available.

## Notes
- The script exits cleanly with a friendly message when OCC is missing.
