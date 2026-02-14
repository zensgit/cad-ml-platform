# DEV_GRAPH2D_STRICT_REGRESSION_EXPERIMENT_20260214

## Goal

Establish a small, repeatable **strict-mode** regression baseline for Graph2D on the local DXF training drawings directory:

- Diagnosis is run in strict mode: **strip DXF text entities + mask filename**
- Compare multiple training setups to see whether strict-mode accuracy moves:
  - baseline (text allowed in training graphs)
  - geometry-only student (strip text entities during training/eval)
  - geometry-only student + distillation (teacher uses titleblock/hybrid)

The intent is not to reach production-level accuracy yet, but to detect whether
changes move the strict-mode metric in the right direction.

## Dataset / Environment

- DXF dir: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`
- Rows in manifest: `110`
- Weak labels: from filename manifest (`scripts/build_dxf_label_manifest.py`, min_confidence=0.8)
- Shared disk graph cache:
  - `/tmp/graph2d_strict_cache_20260214`
- Strict diagnosis enabled via:
  - `scripts/run_graph2d_pipeline_local.py --diagnose-no-text-no-filename`

Common training flags across runs:

- `--epochs 3`
- `--model edge_sage`
- `--loss cross_entropy`
- `--class-weighting inverse`
- `--sampler balanced`
- `--graph-cache both --graph-cache-dir /tmp/graph2d_strict_cache_20260214`
- `--empty-edge-fallback knn --empty-edge-knn-k 8`
- `--diagnose-max-files 80 --seed 42`

## Runs

### Run A: Baseline (Training Graphs Include Text)

Command:

```bash
/usr/bin/time -p .venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --epochs 3 \
  --model edge_sage \
  --loss cross_entropy \
  --class-weighting inverse \
  --sampler balanced \
  --graph-cache both \
  --graph-cache-dir /tmp/graph2d_strict_cache_20260214 \
  --empty-edge-fallback knn \
  --empty-edge-knn-k 8 \
  --diagnose-max-files 80 \
  --diagnose-no-text-no-filename
```

- Work dir: `/tmp/graph2d_pipeline_local_20260214_105307`

### Run B: Geometry-Only Student (Strip Text During Training/Eval)

Command:

```bash
/usr/bin/time -p env DXF_STRIP_TEXT_ENTITIES=true .venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --epochs 3 \
  --model edge_sage \
  --loss cross_entropy \
  --class-weighting inverse \
  --sampler balanced \
  --graph-cache both \
  --graph-cache-dir /tmp/graph2d_strict_cache_20260214 \
  --empty-edge-fallback knn \
  --empty-edge-knn-k 8 \
  --diagnose-max-files 80 \
  --diagnose-no-text-no-filename
```

- Work dir: `/tmp/graph2d_pipeline_local_20260214_105415`

### Run C: Geometry-Only Student + Distill (Teacher = TitleBlock)

Command:

```bash
/usr/bin/time -p env DXF_STRIP_TEXT_ENTITIES=true .venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --epochs 3 \
  --model edge_sage \
  --loss cross_entropy \
  --class-weighting inverse \
  --sampler balanced \
  --graph-cache both \
  --graph-cache-dir /tmp/graph2d_strict_cache_20260214 \
  --empty-edge-fallback knn \
  --empty-edge-knn-k 8 \
  --diagnose-max-files 80 \
  --diagnose-no-text-no-filename \
  --distill \
  --teacher titleblock
```

- Work dir: `/tmp/graph2d_pipeline_local_20260214_105541`

### Run D: Geometry-Only Student + Distill (Teacher = Hybrid)

Command:

```bash
/usr/bin/time -p env DXF_STRIP_TEXT_ENTITIES=true .venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --epochs 3 \
  --model edge_sage \
  --loss cross_entropy \
  --class-weighting inverse \
  --sampler balanced \
  --graph-cache both \
  --graph-cache-dir /tmp/graph2d_strict_cache_20260214 \
  --empty-edge-fallback knn \
  --empty-edge-knn-k 8 \
  --diagnose-max-files 80 \
  --diagnose-no-text-no-filename \
  --distill \
  --teacher hybrid
```

- Work dir: `/tmp/graph2d_pipeline_local_20260214_105630`

## Results (Key Metrics)

All values below are pulled from:

- `eval_metrics.csv` (`__overall__` row) for validation split metrics
- `diagnose/summary.json` for strict-mode diagnosis metrics

| Run | Train Graphs | Distill Teacher | Eval Acc | Eval Top2 | Strict Diagnose Acc | Strict Conf p50 | Strict Conf p90 |
|---|---|---|---:|---:|---:|---:|---:|
| A | text+geom | none | 0.064 | 0.064 | 0.0125 | 0.0272 | 0.0288 |
| B | geom-only | none | 0.021 | 0.043 | 0.0250 | 0.0267 | 0.0296 |
| C | geom-only | titleblock | 0.021 | 0.064 | 0.0250 | 0.0252 | 0.0271 |
| D | geom-only | hybrid | 0.021 | 0.064 | 0.0250 | 0.0252 | 0.0271 |

## Observations

- Strict-mode accuracy is very low in all runs (weak-label truth), with predicted confidences close to a uniform prior (roughly `1/47 ~= 0.021`).
- Geometry-only training slightly improved strict diagnosis accuracy vs baseline (from `0.0125` to `0.025` on the 80-file sample).
- Distillation (titleblock/hybrid teacher) did **not** move strict diagnosis accuracy in this short run (`epochs=3`).

## Notes / Caveats

- These are **small** CPU runs intended to validate wiring and build a stable strict-mode metric, not to claim final model quality.
- Weak labels come from filenames; strict mode deliberately removes filename and DXF text entities from inference, so this evaluation is a proxy for "how much geometry alone matches weak labels."

