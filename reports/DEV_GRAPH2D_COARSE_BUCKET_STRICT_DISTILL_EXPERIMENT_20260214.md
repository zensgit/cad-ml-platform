# DEV_GRAPH2D_COARSE_BUCKET_STRICT_DISTILL_EXPERIMENT_20260214

## Goal

Evaluate whether training Graph2D on **coarse bucket labels** improves strict-mode behavior, and whether **titleblock-based distillation** helps when the student runs in a strict production-like setting:

- Geometry-only graphs (`DXF_STRIP_TEXT_ENTITIES=true`)
- Diagnose in strict mode (`--strip-text-entities --mask-filename`)

## Setup

- DXF dir: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`
- Label normalization:
  - `--normalize-labels` (fine -> bucket)
  - `--clean-min-count 2` (keep buckets with >=2 samples; map others to `other`)
- Resulting buckets (11): `设备`, `传动件`, `罐体`, `轴承件`, `法兰`, `罩盖件`, `过滤组件`, `开孔件`, `支撑件`, `弹簧`, `紧固件`
- Shared disk graph cache: `/tmp/graph2d_strict_cache_20260214`
- Diagnose sample: `--diagnose-max-files 80 --seed 42`

Common training flags:

- `--epochs 5`
- `--model edge_sage`
- `--loss cross_entropy`
- `--class-weighting inverse`
- `--sampler balanced`
- `--empty-edge-fallback knn --empty-edge-knn-k 8`
- Strict diagnose enabled: `--diagnose-no-text-no-filename`

## Runs

### Run A: Coarse Buckets (No Distill)

- Work dir: `/tmp/graph2d_pipeline_local_20260214_110646`

### Run B: Coarse Buckets + Distill (Teacher = TitleBlock, alpha=0.7)

- Work dir: `/tmp/graph2d_pipeline_local_20260214_110759`
- Distill flags:
  - `--distill --teacher titleblock --distill-alpha 0.7`

## Results

Metrics sources:

- `eval_metrics.csv` (`__overall__`) from the pipeline work dir
- `diagnose/summary.json` strict-mode diagnosis

| Run | Eval Acc | Eval Top2 | Strict Diagnose Acc | Strict Conf p50 | Strict Conf p90 | Top Strict Pred |
|---|---:|---:|---:|---:|---:|---|
| A (no distill) | 0.050 | 0.150 | 0.0125 | 0.2731 | 0.3080 | `弹簧` (80/80) |
| B (distill titleblock) | 0.100 | 0.250 | 0.0500 | 0.1260 | 0.1376 | `弹簧` (72/80) |

## Observations

- Coarse-bucket training increased prediction confidence (label_map_size=11), but the strict-mode diagnosis shows **class collapse** (most predictions go to `弹簧`).
- Titleblock distillation (`alpha=0.7`) improved strict diagnosis accuracy (from `0.0125` to `0.05`) and reduced the severity of collapse slightly, but the model is still far from usable in geometry-only strict mode.

## Notes / Caveats

- These runs are CPU-scale and intended for rapid iteration; the strict-mode metric is weak-label based and will be noisy.
- Next likely lever is revisiting imbalance handling for bucket labels (sampler/weights) and increasing epochs, plus feature work (graph construction/sampling) to make geometry-only separable.

