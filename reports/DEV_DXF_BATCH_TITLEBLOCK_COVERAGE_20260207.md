# DEV_DXF_BATCH_TITLEBLOCK_COVERAGE_20260207

## Goal

Since manual visual inspection of drawings is not practical, measure **TitleBlock extraction coverage** over a local DXF corpus by running `/api/v1/analyze` in-process (FastAPI `TestClient`) and emitting aggregated metrics to `summary.json`.

## Changes

- `scripts/batch_analyze_dxf_local.py`
  - Added `titleblock_status` to CSV output.
  - Added `summary["titleblock"]` metrics:
    - `texts_present_*` (raw texts / region entities present)
    - `any_signal_*` (part name / drawing number / material present)
    - `part_name_present_*`
    - `label_present_*`
    - `status_counts` (`matched` / `partial_match` / `no_match`)
    - confidence mean/median

## Verification

### Local Batch (Training corpus, 110 files)

Ran:

```bash
TITLEBLOCK_ENABLED=true PROCESS_FEATURES_ENABLED=false \
  GRAPH2D_ENABLED=false GRAPH2D_FUSION_ENABLED=false FUSION_ANALYZER_ENABLED=false \
  .venv/bin/python scripts/batch_analyze_dxf_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --output-dir reports/experiments/20260207/batch_analyze_dxf_local_titleblock_v1 \
  --seed 2207 --min-confidence 0.8
```

Results (from `summary.json`):

- `total=110`, `success=110`, `error=0`
- TitleBlock coverage:
  - `texts_present_rate=1.0` (110/110)
  - `any_signal_rate=1.0` (110/110)
  - `part_name_present_rate=1.0` (110/110)
  - `label_present_rate=1.0` (110/110)
  - `status_counts={"matched": 108, "partial_match": 2}`
  - `confidence.mean_all=0.845455`, `confidence.median_all=0.85`

Artifacts:

- Summary: `reports/experiments/20260207/batch_analyze_dxf_local_titleblock_v1/summary.json`
- Sanitized CSV: `reports/experiments/20260207/batch_analyze_dxf_local_titleblock_v1/batch_results_sanitized.csv`
- Label distribution: `reports/experiments/20260207/batch_analyze_dxf_local_titleblock_v1/label_distribution.csv`

## Notes / Limits

- This run disables Graph2D/Fusion and process features to isolate TitleBlock behavior.
- Torch is not installed in this environment, so torch-backed classifier paths may log fallback warnings; this does not affect TitleBlock extraction.

