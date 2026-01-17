# DEV_UVNET_GRAPH_DRYRUN_WORKFLOW_20260117

## Summary
Added a GitHub Actions workflow to run the UV-Net graph dry-run on Linux with
micromamba and pythonocc-core.

## Design
- Doc: `docs/UVNET_GRAPH_DRYRUN.md`

## Steps
- Added `.github/workflows/uvnet-graph-dryrun.yml`.
- Attempted to dispatch the workflow via `gh workflow run`, but GitHub reported
  the workflow is not available on the default branch yet.
- Added a push trigger for `feat/l4-uvnet-graph-model` to allow branch runs.

## Results
- Workflow updated with push trigger; pending CI execution.
  Next run uses `micromamba-version: 1.5.8-0`.

## Notes
- The workflow is intended to run in GitHub Actions on Ubuntu to avoid macOS
  pythonocc-core limitations.
