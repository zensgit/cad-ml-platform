# DEV_GRAPH2D_TEMPERATURE_CALIBRATION_20260210

## Goal
Calibrate `GRAPH2D_TEMPERATURE` for the distilled 47-class Graph2D checkpoint so that
**high-confidence geometry-only predictions** (anonymous DXF scenario) are less noisy.

Because temperature scaling does not change argmax (Top-1 accuracy is unchanged), we calibrate for:
- **precision on accepted predictions** (confidence >= threshold), with a minimum accepted count,
instead of minimizing NLL (which tends to favor lower temperatures and higher coverage).

## Changes
- Added `scripts/calibrate_graph2d_temperature.py`
  - grid-search temperatures
  - supports objectives:
    - `precision_at_conf` (default): maximize `accuracy` among samples with `confidence >= thr`
      and accepted `count >= min_count`
    - `nll`: minimize cross-entropy
- `.env.example` now documents:
  - `GRAPH2D_TEMPERATURE`
  - `GRAPH2D_TEMPERATURE_CALIBRATION_PATH`

## Run
Checkpoint:
- `models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth` (47 classes)

Dataset:
- `<DXF_DIR>` (110 DXF; local, not committed)

Manifest (weak labels from filename; gitignored):
- `reports/experiments/20260210/graph2d_distill_titleblock_teacher/manifest_min2.csv`

Command:
```bash
DXF_DIR="<path/to/dxf_dir>"

PYTHONPYCACHEPREFIX=/tmp/pycache \
MPLCONFIGDIR=/tmp/mplconfig \
XDG_CACHE_HOME=/tmp/xdg-cache \
  .venv-graph/bin/python scripts/calibrate_graph2d_temperature.py \
    --manifest reports/experiments/20260210/graph2d_distill_titleblock_teacher/manifest_min2.csv \
    --dxf-dir "$DXF_DIR" \
    --model-path models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth \
    --output-json models/calibration/graph2d_training_dxf_oda_titleblock_distill_20260210_temperature_20260210.json \
    --objective precision_at_conf \
    --objective-confidence-threshold 0.15 \
    --objective-min-count 20 \
    --confidence-thresholds 0.05,0.07,0.1,0.15,0.2 \
    --margin-thresholds 0.0,0.005,0.01,0.02,0.03,0.05 \
    --seed 22
```

Outputs:
- Calibration JSON (no per-file data):
  - `models/calibration/graph2d_training_dxf_oda_titleblock_distill_20260210_temperature_20260210.json`
- Run stdout:
  - `reports/experiments/20260210/graph2d_temperature_calibration/run_stdout.txt`

## Results
Selected temperature (objective: `precision_at_conf`, thr=0.15, min_count=20):
- `temperature = 0.5`

Comparison (same logits, same dataset):
- `T=1.0`:
  - `mean_conf=0.0542`, accepted@0.15=`4` samples, `acc@0.15=0.0`
- `T=0.5`:
  - `mean_conf=0.1136`, accepted@0.15=`22` samples, `acc@0.15=0.5`
- `T=0.25` (best by NLL, but higher-coverage):
  - `mean_conf=0.2506`, accepted@0.15=`60` samples, `acc@0.15=0.3167`

Interpretation:
- For anonymous DXF fine-label gating, `T=0.5` improves “high-confidence precision” vs `T=1.0`,
  while avoiding over-accepting low-margin predictions.
- If the goal is pure probability calibration (NLL), `T=0.25` is better; this is a different objective.

## How To Use
Prefer setting the calibration file (keeps intent explicit and versioned):
```bash
GRAPH2D_TEMPERATURE_CALIBRATION_PATH=models/calibration/graph2d_training_dxf_oda_titleblock_distill_20260210_temperature_20260210.json
```

Or force a temperature directly:
```bash
GRAPH2D_TEMPERATURE=0.5
```
