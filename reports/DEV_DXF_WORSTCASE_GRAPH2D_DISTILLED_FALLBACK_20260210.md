# DEV_DXF_WORSTCASE_GRAPH2D_DISTILLED_FALLBACK_20260210

## Goal
Validate whether a distilled Graph2D checkpoint can provide **non-zero fine-label coverage** in the “worst case” anonymous DXF scenario:
- upload filenames masked (no filename signal),
- titleblock extraction disabled (no titleblock signal),
- process heuristics disabled (no process signal),
so HybridClassifier must rely on **geometry-only** signals.

This is a sanity check for:
- confidence gating strategy (`graph2d_min_conf`), and
- temperature scaling (`GRAPH2D_TEMPERATURE`) for many-class softmax.

## Setup
Checkpoint:
- `models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth` (47-class, EdgeSAGE)

Environment:
- `TITLEBLOCK_ENABLED=false`
- `PROCESS_FEATURES_ENABLED=false`
- `--mask-filename` enabled for the local batch runner

Outputs are aggregated and sanitized:
- Per-file CSV outputs are gitignored by output directory `.gitignore` because they contain original filenames.

## Run
Dataset:
- `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123` (110 DXF)

### Run A: Temperature = 1.0 (baseline for this checkpoint)
```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
MPLCONFIGDIR=/tmp/mplconfig \
XDG_CACHE_HOME=/tmp/xdg-cache \
API_KEY=test \
HYBRID_CLASSIFIER_ENABLED=true \
FILENAME_CLASSIFIER_ENABLED=true \
TITLEBLOCK_ENABLED=false \
TITLEBLOCK_OVERRIDE_ENABLED=false \
PROCESS_FEATURES_ENABLED=false \
GRAPH2D_ENABLED=true \
GRAPH2D_MODEL_PATH=models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth \
GRAPH2D_TEMPERATURE=1.0 \
  .venv-graph/bin/python scripts/batch_analyze_dxf_local.py \
    --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" \
    --output-dir reports/experiments/20260210/batch_analyze_training_dxf_oda_masked_filename_no_titleblock_graph2d_distilled \
    --max-files 110 \
    --seed 22 \
    --min-confidence 0.6 \
    --mask-filename
```

Outputs:
- `reports/experiments/20260210/batch_analyze_training_dxf_oda_masked_filename_no_titleblock_graph2d_distilled/summary.json`
- `reports/experiments/20260210/batch_analyze_training_dxf_oda_masked_filename_no_titleblock_graph2d_distilled/label_distribution.csv`

### Run B: Temperature = 0.5 (more peaky softmax)
```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
MPLCONFIGDIR=/tmp/mplconfig \
XDG_CACHE_HOME=/tmp/xdg-cache \
API_KEY=test \
HYBRID_CLASSIFIER_ENABLED=true \
FILENAME_CLASSIFIER_ENABLED=true \
TITLEBLOCK_ENABLED=false \
TITLEBLOCK_OVERRIDE_ENABLED=false \
PROCESS_FEATURES_ENABLED=false \
GRAPH2D_ENABLED=true \
GRAPH2D_MODEL_PATH=models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth \
GRAPH2D_TEMPERATURE=0.5 \
  .venv-graph/bin/python scripts/batch_analyze_dxf_local.py \
    --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" \
    --output-dir reports/experiments/20260210/batch_analyze_training_dxf_oda_masked_filename_no_titleblock_graph2d_distilled_temp05 \
    --max-files 110 \
    --seed 22 \
    --min-confidence 0.6 \
    --mask-filename
```

Outputs:
- `reports/experiments/20260210/batch_analyze_training_dxf_oda_masked_filename_no_titleblock_graph2d_distilled_temp05/summary.json`
- `reports/experiments/20260210/batch_analyze_training_dxf_oda_masked_filename_no_titleblock_graph2d_distilled_temp05/label_distribution.csv`

## Results Summary
From `summary.json`:

Run A (temp=1.0):
- `graph2d.confidence.p50=0.0406`, `p90=0.0909`
- Hybrid fine-label coverage:
  - `hybrid.label_present_rate=0.2455` (`27/110`)
  - `hybrid.source_counts.graph2d=27`, `fallback=83`
- `soft_override_candidates=2`

Run B (temp=0.5):
- `graph2d.confidence.median=0.0710` (higher than temp=1.0)
- Hybrid fine-label coverage:
  - `hybrid.label_present_rate=0.5909` (`65/110`)
  - `hybrid.source_counts.graph2d=65`, `fallback=45`
- `soft_override_candidates=14`

Label distribution of coarse `part_type` remained stable (expected; governed by L2/Fusion), while the **fine label** (`hybrid.label`) coverage increased substantially.

## Interpretation
1. The distilled checkpoint provides geometry signal, but **confidence is low** in a 47-class setting.
2. `GRAPH2D_TEMPERATURE` materially affects whether predictions pass the Hybrid gating threshold:
   - temp lower than 1.0 increases top-1 probability and unlocks more fine-label coverage.
3. The dynamic Graph2D min-confidence bound (based on `label_map_size`) is necessary to avoid filtering out all geometry-only signals in many-class softmax.

## Next Steps
1. Temperature calibration:
   - Build a small dev calibration set and choose `GRAPH2D_TEMPERATURE` to maximize “high-confidence precision” rather than raw coverage.
2. Add geometry constraints before accepting Graph2D fine labels:
   - Use simple shape priors (e.g., circle/arc counts, border/title-block hints) to reject obviously mismatched fine labels.
3. Distillation improvements:
   - Increase training samples per class (more DXFs or synthetic augmentation).
   - Consider a two-stage model: coarse family first, then fine label within family.

