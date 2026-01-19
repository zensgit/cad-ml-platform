# DEV_UVNET_GRAPH_DRYRUN_GUARD_20260118

## Summary
- Tightened the UV-Net graph dry-run to fail when the batch produces an empty graph.
- Added edge-count logging to the dry-run output.
- Documented the stricter guard in the dry-run guide.

## Changes
- `scripts/train_uvnet_graph_dryrun.py`: fail on 0 nodes or 0 edges; log edge count.
- `docs/UVNET_GRAPH_DRYRUN.md`: noted empty-graph failure and edge-count output.

## Validation
- Attempted to run `scripts/train_uvnet_graph_dryrun.py` in a micromamba Docker container.
- Result: unable to complete validation locally because the Docker daemon was unavailable and the
  earlier micromamba container run hit filesystem write errors when extracting packages.
- Follow-up: rerun the dry-run in CI (workflow) or a local environment with `pythonocc-core`.
