# DEV_UVNET_GRAPH_DRYRUN_WORKFLOW_20260117

## Summary
Added a GitHub Actions workflow to run the UV-Net graph dry-run on Linux with
micromamba and pythonocc-core.

## Design
- Doc: `docs/UVNET_GRAPH_DRYRUN.md`

## Steps
- Added `.github/workflows/uvnet-graph-dryrun.yml`.

## Results
- Workflow added; not executed locally.

## Notes
- The workflow is intended to run in GitHub Actions on Ubuntu to avoid macOS
  pythonocc-core limitations.
