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
- Runs a single batch through the UV-Net model and prints output shapes.

## CI
- Workflow: `.github/workflows/uvnet-graph-dryrun.yml`
- Trigger: manual dispatch, PRs touching UV-Net graph code, or pushes to the
  `feat/l4-uvnet-graph-model` branch.
- The workflow seeds `data/abc_sample` with `tests/fixtures/mock_cube.step` before running.
