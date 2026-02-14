# DEV_GRAPH2D_NODE23_EDGE_SAGE_STRICT_TUNING_20260215

## Goal

Validate whether the newly added optional extra node features (`node_dim > 19`) can
improve strict Graph2D performance when paired with stronger training settings.

Strict mode here means:

- geometry-only student input (`--student-geometry-only`)
- diagnosis with text stripped + filename masked (`--diagnose-no-text-no-filename`)
- weak-label normalization + low-frequency cleaning

## Experiment Setup

Common settings:

- DXF corpus: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf` (110 files)
- `--normalize-labels --clean-min-count 5`
- `--distill --teacher titleblock --distill-alpha 0.1`
- `--dxf-enhanced-keypoints true`
- `--dxf-edge-augment-knn-k 0`
- `--dxf-eps-scale 0.001`
- sampling defaults from pipeline strict path (`frame=0.1`, `long_line=0.4`)

## Runs & Results

| Run | Key Params | Strict Diagnose Accuracy |
|---|---|---|
| A | `model=gcn`, `node_dim=23`, `hidden_dim=64`, `epochs=10`, `seed=42` | `0.1364` |
| B | `model=gcn`, `node_dim=23`, `hidden_dim=128`, `epochs=10`, `seed=42` | `0.2545` |
| C | `model=edge_sage`, `node_dim=23`, `hidden_dim=128`, `epochs=10`, `seed=42` | `0.3545` |
| D (re-check) | `model=edge_sage`, `node_dim=23`, `hidden_dim=128`, `epochs=10`, `seed=7` | `0.3818` |

Reference baseline (previous strict best on this corpus/config family):

- `model=gcn`, `node_dim=19`, strict accuracy `0.2364`

## Artifact Paths

- Run A: `/tmp/graph2d_pipeline_local_20260215_011143`
- Run B: `/tmp/graph2d_pipeline_local_20260215_011345`
- Run C: `/tmp/graph2d_pipeline_local_20260215_011529`
- Run D: `/tmp/graph2d_pipeline_local_20260215_011723`

Each directory contains:

- `pipeline_summary.json`
- `eval_metrics.csv`
- `eval_errors.csv`
- `diagnose/summary.json`
- `graph2d_trained.pth`

## Conclusion

1. `node_dim=23` is not enough by itself (Run A regressed vs baseline).
2. Capacity/model choice matter:
   - Increasing hidden dim to `128` helps (`gcn`: `0.2545`).
   - Switching to `edge_sage` with the same node-dim/capacity yields a large gain (`0.3545~0.3818`).
3. On this dataset and strict protocol, the strongest validated local recipe is:
   - `model=edge_sage`
   - `node_dim=23`
   - `hidden_dim=128`
   - `epochs=10`
   - `distill teacher=titleblock alpha=0.1`

## Suggested Next Step

Promote this as a reproducible strict-training profile (config preset) and run a small
seed sweep (`n>=3`) before replacing any production default model path.

