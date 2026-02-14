# DEV_GRAPH2D_EDGE_AUGMENT_KNN_STRICT_20260214

## Goal
Improve strict-mode Graph2D performance for **geometry-only** DXF graphs by increasing
graph connectivity and reducing isolated nodes:

- Add optional kNN edge augmentation on entity centers (union with epsilon-adjacency).
- Re-run strict diagnosis experiments (`strip DXF text entities` + `masked filename`)
  on the same corpus/config to quantify the effect.

## Changes
### 1) DXF Graph Builder: kNN Edge Augmentation
File: `src/ml/train/dataset_2d.py`

New env var:
- `DXF_EDGE_AUGMENT_KNN_K` (int, default: unset/0)

Behavior:
- When `DXF_EDGE_AUGMENT_KNN_K > 0` and the epsilon-adjacency graph already has edges,
  add kNN edges between entity centers and **union** them with existing edges.
- The augmentation is **not** used to replace the existing empty-edge fallback logic;
  it only applies when adjacency edges exist.

### 2) Wiring / Cache / Tooling
- `scripts/run_graph2d_pipeline_local.py`
  - Added `--dxf-edge-augment-knn-k` (auto default: `8` when `--student-geometry-only`).
  - Records `dxf_edge_augment_knn_k` in `pipeline_summary.json`.
- `scripts/train_2d_graph.py`, `scripts/eval_2d_graph.py`
  - Added `--dxf-edge-augment-knn-k` and env wiring to `DXF_EDGE_AUGMENT_KNN_K`.
- `src/ml/train/dataset_2d.py` (manifest dataset cache key)
  - Added `DXF_EDGE_AUGMENT_KNN_K` to the graph-cache key tokens.
- `scripts/audit_graph2d_strict_graph_quality.py`
  - Records `DXF_EDGE_AUGMENT_KNN_K` and now computes `final_edges` with augmentation.

## Validation
Unit test:
```bash
.venv/bin/pytest tests/unit/test_dataset2d_edge_augment_knn.py -v
```

Syntax checks:
```bash
.venv/bin/python -m py_compile \
  src/ml/train/dataset_2d.py \
  scripts/run_graph2d_pipeline_local.py \
  scripts/train_2d_graph.py \
  scripts/eval_2d_graph.py \
  scripts/audit_graph2d_strict_graph_quality.py
```

## Experiments (Strict Diagnosis)
Corpus:
- `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf` (110 files)

Common setup:
- `--normalize-labels --clean-min-count 5`
- `--student-geometry-only` (strip DXF TEXT/MTEXT/DIMENSION during graph build)
- `--distill --teacher titleblock --distill-alpha 0.1 --distill-temp 3.0`
- `--diagnose-no-text-no-filename` (strict inference)
- kNN edge augmentation: `DXF_EDGE_AUGMENT_KNN_K=8` (auto via pipeline)

Artifacts (local, not committed):
- `/tmp/graph2d_strict_cmp_aug_20260214_154435`

### Results
| Model | strict accuracy | top prediction | notes |
|---|---:|---|---|
| `gcn` | 0.2273 | `传动件` (75/110), `法兰` (32/110) | reduced collapse vs prior baseline |
| `edge_sage` | 0.1909 | `传动件` (110/110) | still fully collapsed |

Baseline comparison (pre-augmentation):
- See `reports/DEV_GRAPH2D_EDGE_SAGE_STRICT_EXPERIMENT_20260214.md`
  - `gcn` strict accuracy: `0.2091`
  - `edge_sage` strict accuracy: `0.1909`

## Graph Quality Audit (With Augmentation)
Command:
```bash
DXF_MAX_NODES=200 \
DXF_SAMPLING_STRATEGY=importance \
DXF_SAMPLING_SEED=42 \
DXF_TEXT_PRIORITY_RATIO=0.0 \
DXF_FRAME_PRIORITY_RATIO=0.1 \
DXF_LONG_LINE_RATIO=0.4 \
DXF_EDGE_AUGMENT_KNN_K=8 \
DXF_EMPTY_EDGE_FALLBACK=knn \
DXF_EMPTY_EDGE_K=8 \
.venv/bin/python scripts/audit_graph2d_strict_graph_quality.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --strip-text-entities
```

Key summary (110 files):
- `fallback_used`: `0/110`
- `adj_edges` p50/p90: ~`258` / `549.2`
- `final_edges` p50/p90: ~`1987` / `2113.8` (after epsilon+knn augmentation)

Artifacts:
- `/tmp/graph2d_graph_audit_aug_20260214_154858`

## Conclusion
Adding kNN edge augmentation (`DXF_EDGE_AUGMENT_KNN_K=8`) improved the `gcn` strict-mode
accuracy from `0.2091` to `0.2273` on this corpus/config and reduced the single-class
collapse slightly. `edge_sage` remains unstable under strict mode and continued to
collapse fully to the majority bucket; it is not recommended as the default strict-mode
model until further model/hyperparam work is done.

