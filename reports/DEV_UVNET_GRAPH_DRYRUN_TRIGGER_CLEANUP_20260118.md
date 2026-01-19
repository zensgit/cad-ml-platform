# DEV_UVNET_GRAPH_DRYRUN_TRIGGER_CLEANUP_20260118

## Summary
- Removed the push trigger from the UV-Net graph dry-run workflow.
- Kept manual dispatch and PR-only execution to reduce accidental runs.

## Changes
- `.github/workflows/uvnet-graph-dryrun.yml`: removed the `push` trigger block.
- `docs/UVNET_GRAPH_DRYRUN.md`: updated trigger description.

## Validation
- Configuration change verified by inspection (no runtime execution required).
