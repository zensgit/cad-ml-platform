# DEV_DXF_BATCH_ANALYZE_HYBRID_BASELINE_20260209

## Goal
Establish a non-interactive baseline for DXF fine-label classification using the in-process `/api/v1/analyze/` pipeline (FastAPI TestClient), so we can validate accuracy signals without manually opening drawings.

This run focuses on:
- `HybridClassifier` decision behavior (`filename` / `titleblock` / `graph2d` / `process`)
- Field population and confidence distributions
- Aggregated label distribution (no per-file review required)

## Run
Environment:
- Used `.venv-graph` (torch-enabled) so Graph2D and any torch-backed classifiers are available.
- Set `DISABLE_MODEL_SOURCE_CHECK=True` to avoid slow network hoster checks during local evaluation.

Command:
```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
MPLCONFIGDIR=/tmp/mplconfig \
XDG_CACHE_HOME=/tmp/xdg-cache \
DISABLE_MODEL_SOURCE_CHECK=True \
API_KEY=test \
HYBRID_CLASSIFIER_ENABLED=true \
FILENAME_CLASSIFIER_ENABLED=true \
TITLEBLOCK_ENABLED=true \
PROCESS_FEATURES_ENABLED=true \
GRAPH2D_ENABLED=true \
GRAPH2D_FUSION_ENABLED=true \
FUSION_ANALYZER_ENABLED=true \
  .venv-graph/bin/python scripts/batch_analyze_dxf_local.py \
    --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" \
    --output-dir reports/experiments/20260209/batch_analyze_training_dxf_oda \
    --max-files 200 \
    --seed 22 \
    --min-confidence 0.6
```

Outputs:
- Aggregated:
  - `reports/experiments/20260209/batch_analyze_training_dxf_oda/summary.json`
  - `reports/experiments/20260209/batch_analyze_training_dxf_oda/label_distribution.csv`
- Per-file outputs (gitignored by default; can include file names):
  - `reports/experiments/20260209/batch_analyze_training_dxf_oda/batch_results*.csv`
  - `reports/experiments/20260209/batch_analyze_training_dxf_oda/batch_low_confidence*.csv`

## Results (110 DXF)
From `summary.json`:
- Total: `110`
- Success: `110`
- Error: `0`
- Overall confidence buckets:
  - `gte_0_8`: `110`
- Filename classifier:
  - `label_present_rate`: `1.0` (110/110)
  - `match_type_counts`: `exact=110`
  - confidence mean/median: `0.95 / 0.95`
- Hybrid decision:
  - `source_counts`: `filename=110`
  - confidence mean/median: `0.95 / 0.95`
- Graph2D:
  - `status_counts`: `ok=110`
  - confidence mean/median: `0.345 / 0.315`
- Titleblock:
  - `status_counts`: `matched=108`, `partial_match=2`
  - `part_name_present_rate`: `1.0`
  - `label_present_rate`: `1.0`
  - confidence mean/median: `0.845 / 0.85`

From `label_distribution.csv`:
- Top labels (count):
  - `盖=11`, `轴承=6`, `挡板=5`, `罩=4`, `支腿=4`, `旋转组件=4`, `过滤托架=4`, `底板=4` ...

## Notes
- This baseline helps explain the earlier observation where `Graph2D` alone looked wrong/low-confidence: in this dataset the system can derive strong labels from filename + titleblock, so `HybridClassifier` correctly prefers those sources.
- Running without a torch environment is still acceptable: `filename/titleblock/process` remain available, while `graph2d` gracefully degrades as `model_unavailable` when torch/checkpoints are missing.

