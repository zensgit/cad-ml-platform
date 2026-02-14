# DEV_GRAPH2D_ENHANCED_KEYPOINTS_EDGE_AUGMENT_STRATEGY_20260214

## Goal

Improve Graph2D strict-mode (geometry-only) behavior on real DXF drawings by:

- building a more faithful epsilon-adjacency graph for circles/arcs, and
- making kNN edge augmentation configurable (to avoid adding noisy edges by default).

## Scope / Dataset

- DXF corpus: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf` (110 files)
- Strict diagnose mode:
  - strip DXF text entities (`--strip-text-entities`)
  - mask filename (`--mask-filename`)
  - ground-truth labels from manifest CSV (weak labels from filename; normalized + cleaned)

## Changes

### 1) Enhanced epsilon-adjacency keypoints (feature-flagged)

When `DXF_ENHANCED_KEYPOINTS=true`:

- `CIRCLE`: add 4 cardinal points on the circumference (in addition to the center).
- `ARC`: add a mid-arc point keypoint (in addition to endpoints).

Files:

- `src/ml/train/dataset_2d.py` (graph build)
- `scripts/audit_graph2d_strict_graph_quality.py` (audit parity)
- `scripts/run_graph2d_pipeline_local.py` (CLI wiring)
- `scripts/train_2d_graph.py` / `scripts/eval_2d_graph.py` (CLI wiring)

### 2) kNN augmentation strategy (feature-flagged)

New env var:

- `DXF_EDGE_AUGMENT_STRATEGY=union_all|isolates_only`

Behavior (when `DXF_EDGE_AUGMENT_KNN_K>0` and epsilon-adjacency is non-empty):

- `union_all` (default): add kNN edges for every node (existing behavior).
- `isolates_only`: add kNN edges only for nodes that are isolated (degree 0) in the epsilon graph.

Files:

- `src/ml/train/dataset_2d.py`
- `scripts/audit_graph2d_strict_graph_quality.py`
- `scripts/run_graph2d_pipeline_local.py`
- `scripts/train_2d_graph.py` / `scripts/eval_2d_graph.py`

Pipeline heuristic update (`scripts/run_graph2d_pipeline_local.py`):

- For `--student-geometry-only` + enhanced keypoints enabled, default `DXF_EDGE_AUGMENT_KNN_K=0`.
- Otherwise (geometry-only with enhanced keypoints disabled), default `DXF_EDGE_AUGMENT_KNN_K=8`.

## Verification

Unit tests:

```bash
./.venv/bin/python -m pytest -q \
  tests/unit/test_dataset2d_edge_augment_knn.py \
  tests/unit/test_dataset2d_enhanced_keypoints.py \
  tests/unit/test_dataset2d_edge_augment_strategy.py
```

## Experiments (Strict Diagnose Accuracy)

Common configuration:

- `--student-geometry-only`
- `--normalize-labels --clean-min-count 5`
- `--distill --teacher titleblock --distill-alpha 0.1`
- `--diagnose-no-text-no-filename`
- `--epochs 3 --batch-size 4`
- `--graph-cache disk`

### Run A: Enhanced keypoints, no kNN augmentation

```bash
./.venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --work-dir /tmp/graph2d_isolate_aug_cmp_20260214_a_k0 \
  --student-geometry-only \
  --diagnose-no-text-no-filename \
  --dxf-enhanced-keypoints true \
  --dxf-edge-augment-knn-k 0 \
  --dxf-edge-augment-strategy union_all \
  --normalize-labels --clean-min-count 5 \
  --distill --teacher titleblock --distill-alpha 0.1 \
  --epochs 3 --batch-size 4 \
  --graph-cache disk
```

- Strict accuracy (`diagnose/summary.json -> accuracy`): `0.2364`

### Run B: Enhanced keypoints + kNN augmentation (union_all)

```bash
./.venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --work-dir /tmp/graph2d_isolate_aug_cmp_20260214_b_unionall_k8 \
  --student-geometry-only \
  --diagnose-no-text-no-filename \
  --dxf-enhanced-keypoints true \
  --dxf-edge-augment-knn-k 8 \
  --dxf-edge-augment-strategy union_all \
  --normalize-labels --clean-min-count 5 \
  --distill --teacher titleblock --distill-alpha 0.1 \
  --epochs 3 --batch-size 4 \
  --graph-cache disk
```

- Strict accuracy: `0.2273`

### Run C: Enhanced keypoints + kNN augmentation (isolates_only)

```bash
./.venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --work-dir /tmp/graph2d_isolate_aug_cmp_20260214_c_isolatesonly_k8 \
  --student-geometry-only \
  --diagnose-no-text-no-filename \
  --dxf-enhanced-keypoints true \
  --dxf-edge-augment-knn-k 8 \
  --dxf-edge-augment-strategy isolates_only \
  --normalize-labels --clean-min-count 5 \
  --distill --teacher titleblock --distill-alpha 0.1 \
  --epochs 3 --batch-size 4 \
  --graph-cache disk
```

- Strict accuracy: `0.2182`

## Conclusion

- On this corpus/config, `DXF_ENHANCED_KEYPOINTS=true` improves strict accuracy.
- With enhanced keypoints enabled, adding kNN augmentation degraded strict accuracy in both tested strategies.
- Recommended strict/geometry-only setting for this corpus/config:
  - `DXF_ENHANCED_KEYPOINTS=true`
  - `DXF_EDGE_AUGMENT_KNN_K=0`

