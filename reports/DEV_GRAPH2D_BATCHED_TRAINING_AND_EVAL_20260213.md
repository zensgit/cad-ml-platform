# DEV_GRAPH2D_BATCHED_TRAINING_AND_EVAL_20260213

## Goal

Speed up local Graph2D training/evaluation loops by batching multiple DXF graphs per step without adding a `torch_geometric` dependency, while keeping the inference API backward-compatible.

## Changes

### 1) Batched Pooling Support in 2D Graph Models

File: `src/ml/train/model_2d.py`

- Added `_global_mean_pool(x, batch)` helper.
- Extended model forwards to accept an optional `batch` vector:
  - `SimpleGraphClassifier.forward(..., batch=None)`
  - `EdgeGraphSageClassifier.forward(..., batch=None)`
- When `batch` is provided, pooling is performed per-graph and the model returns logits shaped `(B, C)`.
- When `batch` is omitted, behavior remains unchanged and the model returns logits shaped `(1, C)` (used by `src/ml/vision_2d.py` inference).

### 2) Batched Training Loop (Concatenate Graphs + Shift Edges)

File: `scripts/train_2d_graph.py`

- Reworked `_collate(...)` to:
  - Filter empty graphs.
  - Concatenate node features.
  - Shift edge indices by node offsets.
  - Produce a `batch` vector for pooling.
- Training now performs one forward/backward per batch instead of iterating per-graph.
- Implemented `--downweight-label/--downweight-factor` by adjusting class-weight tensors when supported by the selected loss.

### 3) Batched Evaluation Loop

File: `scripts/eval_2d_graph.py`

- Reworked `_collate(...)` and evaluation loop to run logits/softmax/top-k on batched graphs.

### 4) Unit Tests

File: `tests/unit/test_graph2d_batched_pooling.py`

- Verifies that batching two disconnected graphs (with a `batch` vector) produces logits identical to running each graph independently.
- Covers both `SimpleGraphClassifier` and `EdgeGraphSageClassifier`.
- Skips when `torch` is unavailable.

## Validation

### Static Checks

- `.venv/bin/python -m py_compile src/ml/train/model_2d.py scripts/train_2d_graph.py scripts/eval_2d_graph.py` (passed)
- `.venv/bin/python -m flake8 scripts/train_2d_graph.py scripts/eval_2d_graph.py src/ml/train/model_2d.py tests/unit/test_graph2d_batched_pooling.py --max-line-length=100` (passed)

### Unit Tests

- `.venv/bin/python -m pytest tests/unit/test_graph2d_batched_pooling.py tests/unit/test_graph2d_script_config.py -q` (passed)

### Pipeline Smoke

Command:

```bash
/usr/bin/time -p .venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --normalize-labels \
  --clean-min-count 2 \
  --model edge_sage \
  --loss cross_entropy \
  --class-weighting inverse \
  --sampler balanced \
  --epochs 1 \
  --max-samples 40 \
  --diagnose-max-files 20 \
  --graph-cache both \
  --empty-edge-fallback knn \
  --empty-edge-knn-k 8
```

Observed:

- Completed successfully (work dir: `/tmp/graph2d_pipeline_local_20260213_231242`)
- Timing: `real 21.15s`

