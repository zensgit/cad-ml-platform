# DEV_UVNET_GRAPH_DRYRUN_CI_ATTEMPT_20260117

## Summary
Triggered the UV-Net graph dry-run workflow on the feature branch; the run failed
before environment setup due to an invalid micromamba version format.

## Steps
- Triggered: `gh workflow run uvnet-graph-dryrun.yml --ref feat/l4-uvnet-graph-model`.
- Observed run: `gh run view 21096516600 --log`.

## Results
- `setup-micromamba` failed: `micromamba-version must be either 'latest' or a version matching 1.2.3-0`.
- No dry-run log artifact produced.

## Notes
- Workflow updated to use `micromamba-version: 1.5.8-0` for the next run.
