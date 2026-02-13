# DEV_GRAPH2D_COARSE_LOSS_STRATEGY_COMPARISON_20260213

## Goal

Compare coarse-bucket Graph2D training strategies on the same DXF dataset to decide which setup is worth iterating further, after the batched training/eval speedup landed.

Dataset:

- `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf` (110 DXFs)
- Labels: filename weak labels -> normalized/cleaned into 11 coarse buckets (`--normalize-labels --clean-min-count 2`)

Common settings:

- `model=edge_sage`
- `sampler=balanced`
- `epochs=15`
- Sampling: `DXF_MAX_NODES=200`, `DXF_SAMPLING_STRATEGY=importance`, `DXF_SAMPLING_SEED=42`, `DXF_TEXT_PRIORITY_RATIO=0.3`
- Graph build: `--empty-edge-fallback knn --empty-edge-knn-k 8`
- Cache: `--graph-cache both`
- Diagnose: `--diagnose-max-files 80` with `--manifest-csv` truth mode

## Runs

### 1) Baseline: `cross_entropy` + inverse weights

Work dir: `/tmp/graph2d_pipeline_local_20260213_233141`

- Eval (val split): `acc=0.350`, `top2=0.600`, `macro_f1=0.311`, `weighted_f1=0.285`
- Diagnose (manifest truth, sampled 80): `accuracy=0.275`
- Time: `real 43.07s`

Report: `reports/DEV_GRAPH2D_LOCAL_RETRAIN_COARSE_BATCHED_20260213.md`

### 2) Downweight experiment: `cross_entropy` + inverse weights + `--downweight-label 紧固件`

Work dir: `/tmp/graph2d_downweight_20260213_233520`

- Eval (val split): `acc=0.300`, `top2=0.500`, `macro_f1=0.300`, `weighted_f1=0.251`
- Diagnose (manifest truth, sampled 80): `accuracy=0.225`

Report: `reports/DEV_GRAPH2D_DOWNWEIGHT_LABEL_EXPERIMENT_20260213.md`

### 3) Focal loss: `focal` + inverse weights

Work dir: `/tmp/graph2d_pipeline_local_20260213_233904`

- Eval (val split): `acc=0.250`, `top2=0.300`, `macro_f1=0.235`, `weighted_f1=0.129`
- Diagnose (manifest truth, sampled 80): `accuracy=0.150`
- Time: `real 35.58s`

### 4) Logit adjustment: `logit_adjusted` (no class weights)

Work dir: `/tmp/graph2d_pipeline_local_20260213_234006`

- Eval (val split): `acc=0.400`, `top2=0.500`, `macro_f1=0.325`, `weighted_f1=0.314`
- Diagnose (manifest truth, sampled 80): `accuracy=0.2625`
- Time: `real 35.83s`

## Recommendation

- **Keep the current baseline (`cross_entropy` + inverse weights + balanced sampler) as the default coarse-bucket iteration loop**, because it remains best on the larger manifest-truth diagnose sample (`0.275` vs `0.2625`).
- `logit_adjusted` is a reasonable alternative to keep around for follow-up, since it improved the tiny 20-sample val split (`acc=0.400`) while staying close on diagnose.
- `focal` and the `紧固件` downweight experiment did not improve accuracy in this dataset and are not recommended as next defaults.

## Next Step Candidates (beyond loss tweaks)

- Increase training data (the current set is small and noisy labels are filename-derived).
- Evaluate distillation / multi-signal training focused on the “no-filename” scenario (mask filename teacher input and rely on titleblock/process signals).

