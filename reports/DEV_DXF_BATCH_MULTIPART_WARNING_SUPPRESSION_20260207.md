# DEV_DXF_BATCH_MULTIPART_WARNING_SUPPRESSION_20260207

## Goal

Reduce noisy `python_multipart.multipart` warnings during local DXF batch evaluation runs (`scripts/batch_analyze_dxf_local.py`) that upload files via FastAPI `TestClient`.

These warnings were observed as:

- `python_multipart.multipart` `Skipping data after last boundary`

They are not actionable for this workflow and can dominate the console output.

## Changes

- `scripts/batch_analyze_dxf_local.py`
  - Set logger levels to suppress multipart parser warnings:
    - `python_multipart.multipart` -> `ERROR`
    - `python_multipart` -> `ERROR`

## Verification

### Local Smoke Batch (2 files)

Ran:

```bash
GRAPH2D_ENABLED=false GRAPH2D_FUSION_ENABLED=false FUSION_ANALYZER_ENABLED=false \
  .venv/bin/python scripts/batch_analyze_dxf_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --output-dir reports/experiments/20260207/batch_analyze_dxf_local_smoke_v1 \
  --max-files 2 --seed 22
```

Results:

- No `python_multipart` warnings observed in stdout/stderr.
- Summary: `reports/experiments/20260207/batch_analyze_dxf_local_smoke_v1/summary.json`

## Notes / Limits

- Torch is not installed in this environment; any torch-backed classifier components may log fallback warnings. This change only targets multipart parsing noise.

