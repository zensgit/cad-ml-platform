# DEV_UVNET_GRAPH_DRYRUN_CI_DATA_20260117

## Summary
The UV-Net graph dry-run workflow completed with pythonocc-core installed, but
failed to find `data/abc_sample` on the runner.

## Steps
- Observed run: `gh run view 21096649015 --log`.

## Results
- Dry-run exited early: `Data directory not found: data/abc_sample`.

## Fix
- Added `tests/fixtures/mock_cube.step` and updated the workflow to copy it into
  `data/abc_sample` before running the dry-run.

## Notes
- Re-run the workflow to confirm the forward pass executes against the sample STEP file.
