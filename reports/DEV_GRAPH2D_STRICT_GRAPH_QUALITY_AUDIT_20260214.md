# DEV_GRAPH2D_STRICT_GRAPH_QUALITY_AUDIT_20260214

## Goal
Audit Graph2D **strict-mode** graph construction quality on a local DXF corpus:
- strip DXF text/annotation entities
- importance sampling enabled (`DXF_MAX_NODES=200`)
- edges built via epsilon adjacency over entity keypoints (dataset logic)

This is meant to answer:
- Do we frequently hit empty-edge fallback?
- Are graphs sparse/dense? (edge density)
- Are we truncating too aggressively (node counts hitting max)?

## Method
Used a dedicated audit script:
- `scripts/audit_graph2d_strict_graph_quality.py`

The audit reproduces the dataset graph-building logic (keypoints + epsilon adjacency) and reports:
- node counts
- adjacency edge counts
- connected components (on final edges)
- whether empty-edge fallback would have been used

## Configuration
Environment used for both runs:
- `DXF_MAX_NODES=200`
- `DXF_SAMPLING_STRATEGY=importance`
- `DXF_SAMPLING_SEED=42`
- `DXF_TEXT_PRIORITY_RATIO=0.0`
- `DXF_FRAME_PRIORITY_RATIO=0.1`
- `DXF_LONG_LINE_RATIO=0.4`

Corpus:
- `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf` (110 files)

## Runs
### Run A: fully_connected fallback (baseline)
```bash
DXF_EMPTY_EDGE_FALLBACK=fully_connected DXF_EMPTY_EDGE_K=8 \
.venv/bin/python scripts/audit_graph2d_strict_graph_quality.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --strip-text-entities \
  --max-files 110 \
  --seed 42 \
  --output-dir /tmp/graph2d_graph_audit_20260214_151951_fc
```

Summary:
- `fallback_used`: `0/110` (0%)
- `nodes`: min `13`, p50 `200`, p90 `200`, max `200`
- `adj_edges`: p50 `258`, p90 `549.2`

### Run B: knn fallback
```bash
DXF_EMPTY_EDGE_FALLBACK=knn DXF_EMPTY_EDGE_K=8 \
.venv/bin/python scripts/audit_graph2d_strict_graph_quality.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --strip-text-entities \
  --max-files 110 \
  --seed 42 \
  --output-dir /tmp/graph2d_graph_audit_20260214_151951_knn
```

Summary:
- `fallback_used`: `0/110` (0%)
- `nodes`: min `13`, p50 `200`, p90 `200`, max `200`
- `adj_edges`: p50 `258`, p90 `549.2`

## Findings
1. **Empty-edge fallback was never used** on this corpus under strict-mode graph build.
   - This means the current strict-mode collapse is not due to the graph being completely edgeless.

2. **Sampling hits the max node cap heavily** (p50/p90 both 200).
   - Many graphs are being truncated to 200 sampled entities, so sampling quality matters.

3. **Graphs are sparse** even after sampling.
   - With `n=200`, density is typically ~`258 / (200*199) ≈ 0.0065` (directed).

## Implications / Next Steps
- Since fallback is not used, switching empty-edge fallback (fully_connected vs kNN) will not change graphs for this corpus.
- The next likely wins are:
  - improve edge construction (e.g. augment epsilon adjacency with kNN edges to improve message passing locality),
  - run EdgeSage with `edge_attr` once edges are augmented,
  - enrich node features for arcs/polylines/geometry patterns.
