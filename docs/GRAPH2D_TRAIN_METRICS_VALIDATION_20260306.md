# Graph2D Train Metrics Validation

## Scope

Branch: `feat/graph2d-train-metrics`

Commit:
- `d17ca87` `feat: emit graph2d training metrics artifacts`

## What Changed

- Upgraded default Graph2D training config.
- Added `--metrics-out` support to `scripts/train_2d_graph.py`.
- Emitted structured training metrics artifacts with:
  - best epoch
  - final validation accuracy
  - final loss
  - epochs run
  - class stats
  - sampling overrides
  - epoch history

## Key Files

- `config/graph2d_training.yaml`
- `scripts/train_2d_graph.py`
- `tests/unit/test_train_2d_graph_metrics_artifact.py`

## Validation

Commands:

```bash
python3 -m pytest -q tests/unit/test_train_2d_graph_metrics_artifact.py

flake8 \
  scripts/train_2d_graph.py \
  tests/unit/test_train_2d_graph_metrics_artifact.py \
  --max-line-length=100

python3 -m py_compile \
  scripts/train_2d_graph.py \
  tests/unit/test_train_2d_graph_metrics_artifact.py
```

Results:

- `1 passed`
- `flake8` passed
- `py_compile` passed

## Risks

- Training defaults now favor `focal + sqrt + balanced`, which can change model behavior and training curves.
