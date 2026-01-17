#!/usr/bin/env markdown
# UV-Net Checkpoint Inspector

## Goal
Provide a lightweight script to verify checkpoint integrity and run a minimal
forward pass without spinning up the full training pipeline.

## Usage
```
source .venv-graph/bin/activate
python3 scripts/uvnet_checkpoint_inspect.py --path models/smoke_test_model.pth
```

Make target:
```
make uvnet-checkpoint-inspect UVNET_CHECKPOINT=models/smoke_test_model.pth PYTHON=.venv-graph/bin/python
```

## Output
- Prints the checkpoint config.
- Runs a single forward pass on a small synthetic graph.
- Reports logits and embedding shapes.
