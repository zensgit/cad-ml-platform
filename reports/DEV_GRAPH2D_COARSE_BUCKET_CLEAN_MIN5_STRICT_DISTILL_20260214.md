# DEV_GRAPH2D_COARSE_BUCKET_CLEAN_MIN5_STRICT_DISTILL_20260214

## Goal

Continue strict-mode iteration for Graph2D by reducing the bucket label-space further (merge low-frequency buckets into `其他`) and re-evaluating whether titleblock distillation improves geometry-only strict inference:

- Student graphs: `DXF_STRIP_TEXT_ENTITIES=true` (geometry-only)
- Diagnose: strict mode (strip DXF text entities + mask filename)

This is an extension of the earlier coarse-bucket runs, focusing on mitigating class collapse by increasing `--clean-min-count`.

## Setup

- DXF dir: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`
- Label pipeline:
  - `--normalize-labels` (fine -> bucket)
  - `--clean-min-count 5` (map buckets with <5 samples into `其他`)
- Shared disk graph cache: `/tmp/graph2d_strict_cache_20260214`
- Diagnose sample: `--diagnose-max-files 80 --seed 42`
- Strict mode: `--diagnose-no-text-no-filename` (passes `--strip-text-entities --mask-filename`)

Common training flags:

- `--epochs 5`
- `--model edge_sage`
- `--loss cross_entropy`
- `--class-weighting inverse`
- `--sampler balanced`
- `--empty-edge-fallback knn --empty-edge-knn-k 8`

## Runs

### Run A: Clean>=5 (No Distill)

- Work dir: `/tmp/graph2d_pipeline_local_20260214_111139`

### Run B: Clean>=5 + Distill (Teacher = TitleBlock, alpha=0.7)

- Work dir: `/tmp/graph2d_pipeline_local_20260214_111228`
- Distill flags:
  - `--distill --teacher titleblock --distill-alpha 0.7`

## Results

Metrics sources:

- `eval_metrics.csv` (`__overall__`) from the pipeline work dir
- `diagnose/summary.json` strict-mode diagnosis

| Run | Eval Acc | Eval Top2 | Strict Diagnose Acc | Strict Conf p50 | Strict Conf p90 | Top Strict Pred |
|---|---:|---:|---:|---:|---:|---|
| A (no distill) | 0.167 | 0.278 | 0.1125 | 0.1616 | 0.2172 | `过滤组件` (71/80) |
| B (distill titleblock) | 0.222 | 0.333 | 0.1750 | 0.1617 | 0.1743 | `传动件` (77/80) |

## Observations

- Increasing `--clean-min-count` from `2` to `5` meaningfully improved strict-mode accuracy on the sampled 80 files:
  - `0.0125` (clean>=2, no distill) -> `0.1125` (clean>=5, no distill)
  - `0.0500` (clean>=2, distill) -> `0.1750` (clean>=5, distill)
- Distillation improved strict-mode accuracy further in this configuration (+0.0625 absolute), but predictions are still heavily collapsed into a single bucket (`传动件`).

## Notes / Caveats

- This is still weak-label evaluation and not a production-quality metric; it is used to track strict-mode iteration directionally.
- The remaining collapse suggests geometry-only features are not yet sufficiently discriminative; next steps likely require feature/graph improvements or different imbalance handling.

