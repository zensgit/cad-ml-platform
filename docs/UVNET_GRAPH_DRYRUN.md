#!/usr/bin/env markdown
# UV-Net Graph Dry-Run

## Goal
Provide a lightweight script to load STEP-based graph data and execute a single
forward pass through the UV-Net model.

## Usage
```
source .venv-graph/bin/activate
python3 scripts/train_uvnet_graph_dryrun.py --data-dir data/abc_subset
```

## Behavior
- Skips execution with a friendly message if `pythonocc-core` is unavailable.
- Loads graph data from `ABCDataset(output_format="graph")`.
- Fails the run if the batch produces an empty graph (0 nodes or 0 edges).
- Runs a single batch through the UV-Net model and prints output shapes plus
  node/edge counts.

## CI
- Workflow: `.github/workflows/uvnet-graph-dryrun.yml`
- Trigger: manual dispatch or PRs touching UV-Net graph code.
- The workflow seeds `data/abc_sample` with `tests/fixtures/*.step` before running
  (cube + additional fixtures such as `eight_cyl.stp` and `as1_oc_214.stp` from
  pythonocc-core test data).
