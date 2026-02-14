# DEV_GRAPH2D_EDGE_SAGE_STRICT_EXPERIMENT_20260214

## Goal
Compare `gcn` vs `edge_sage` under **strict** diagnosis conditions:

- DXF text/annotation entities stripped at inference
- Filename masked (`masked.dxf`) at inference
- Student graphs built in **geometry-only** mode (`DXF_STRIP_TEXT_ENTITIES=true`)
- Labels normalized into coarse buckets, then cleaned with `min_count=5`

This is intended to answer: does EdgeSage (edge_attr-aware) mitigate the strict-mode
collapse observed with geometry-only graphs?

## Dataset / Environment
- Corpus: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf` (110 DXF files)
- Python: `.venv/bin/python` (Python 3.11.13)
- Branch: `main`
- Artifacts (local, not committed): `/tmp/graph2d_strict_cmp_20260214_153010`

## Commands
GCN run:
```bash
.venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --work-dir "/tmp/graph2d_strict_cmp_20260214_153010/gcn" \
  --graph-cache both --graph-cache-dir "/tmp/graph2d_strict_cmp_20260214_153010/graph_cache" \
  --epochs 3 --batch-size 4 --hidden-dim 64 --lr 1e-3 \
  --model gcn --loss focal --class-weighting sqrt --sampler balanced \
  --normalize-labels --clean-min-count 5 \
  --student-geometry-only \
  --distill --teacher titleblock --distill-alpha 0.1 --distill-temp 3.0 \
  --diagnose-no-text-no-filename
```

EdgeSage run:
```bash
.venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --work-dir "/tmp/graph2d_strict_cmp_20260214_153010/edge_sage" \
  --graph-cache both --graph-cache-dir "/tmp/graph2d_strict_cmp_20260214_153010/graph_cache" \
  --epochs 3 --batch-size 4 --hidden-dim 64 --lr 1e-3 \
  --model edge_sage --loss focal --class-weighting sqrt --sampler balanced \
  --normalize-labels --clean-min-count 5 \
  --student-geometry-only \
  --distill --teacher titleblock --distill-alpha 0.1 --distill-temp 3.0 \
  --diagnose-no-text-no-filename
```

## Label Buckets (After Normalize + Clean>=5)
The manifest normalized to 11 coarse buckets, then cleaned to 7 buckets + `其他`
(`label_map_size=8`), with top true-label counts:

- `传动件`: 21
- `设备`: 19
- `罐体`: 18
- `其他`: 14
- `法兰`: 11
- `轴承件`: 11
- `过滤组件`: 8
- `罩盖件`: 8

## Results
### Strict Diagnosis (strip DXF text + masked filename)
From:
- `.../gcn/diagnose/summary.json`
- `.../edge_sage/diagnose/summary.json`

| Model | strict accuracy | conf p50 | conf p90 | top prediction |
|---|---:|---:|---:|---|
| `gcn` | 0.2091 | 0.1424 | 0.1481 | `传动件` (88/110) |
| `edge_sage` | 0.1909 | 0.1544 | 0.1652 | `传动件` (110/110) |

### Eval Split (non-strict, manifest split)
Both runs reported the same evaluation split accuracy:
- `Validation samples=18 acc=0.222 top2=0.333`

## Findings
1. **EdgeSage did not improve strict performance** on this corpus/config; it collapsed harder
   (predicting `传动件` for all 110 files).
2. Confidence remained **very low** (p50 ~0.14-0.16), consistent with weak geometry-only
   separability for these labels.

## Conclusion / Next Step
Given strict-mode collapse persists and the earlier graph-build audit showed **empty-edge
fallback is rarely used** (0/110), the next step is to improve the *graph signal*:

- Add an **optional kNN edge augmentation** pass (union with existing epsilon-adjacency
  edges), controlled via an environment variable/flag.
- Re-run the same strict-mode comparison to confirm whether augmented connectivity
  improves strict accuracy and reduces single-class collapse.

