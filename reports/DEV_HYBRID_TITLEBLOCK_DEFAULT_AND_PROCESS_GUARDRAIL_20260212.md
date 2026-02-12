# DEV_HYBRID_TITLEBLOCK_DEFAULT_AND_PROCESS_GUARDRAIL_20260212

Date: 2026-02-12

## Summary

Improved DXF fine-label classification when filenames are not informative by enabling
TitleBlock extraction by default, tightening the TitleBlock override gate, and preventing
ProcessClassifier drawing-type labels (e.g. `零件图/装配图`) from competing with part-name
labels in HybridClassifier fusion.

## Changes

- Updated `config/hybrid_classifier.yaml`
  - Enabled `titleblock.enabled: true` by default (config version `1.1.1`).
  - Kept `titleblock.override_enabled: false` by default (can still be enabled via env).

- Updated `src/ml/hybrid_classifier.py`
  - TitleBlock override gate now uses `filename_conf < filename_min_conf` instead of a
    hard-coded `0.5`.
  - ProcessClassifier drawing-type labels no longer participate in fusion when a part-name
    label is present (from filename/titleblock/graph2d). This avoids falling into low-score
    fusion mode when filename is masked and TitleBlock is strong.

- Updated `scripts/batch_analyze_dxf_local.py`
  - Summary now reports effective TitleBlock settings (config + env), including:
    - `titleblock.enabled`
    - `titleblock.override_enabled`
    - `titleblock.min_confidence_effective`

## Verification

- Unit/integration tests:
  - `.venv/bin/python -m pytest tests/unit/test_hybrid_classifier_coverage.py tests/unit/test_hybrid_config_loader.py tests/integration/test_analyze_dxf_fusion.py -v`
  - Result: passed

- Repo validation:
  - `make validate-core-fast`
  - Result: passed

## Local Evaluation (DXF Training Set Batch)

Ran `scripts/batch_analyze_dxf_local.py` against a local DXF directory:
- With original filenames
- With masked filenames (simulates production where filenames carry no semantic label)

Artifacts (aggregated only):
- With filename:
  - `reports/experiments/20260212/batch_analyze_training_dxf/titleblock_default/with_filename/summary.json`
  - `reports/experiments/20260212/batch_analyze_training_dxf/titleblock_default/with_filename/label_distribution.csv`
- Masked filename:
  - `reports/experiments/20260212/batch_analyze_training_dxf/titleblock_default/masked_filename/summary.json`
  - `reports/experiments/20260212/batch_analyze_training_dxf/titleblock_default/masked_filename/label_distribution.csv`

Key results (from `summary.json`):
- With filename:
  - Filename label present rate: `1.0` (exact matches)
  - Hybrid source: `filename` (30/30)
  - TitleBlock matched: `30/30` (mean confidence `0.85`)
- Masked filename:
  - Filename label present rate: `0.0` (expected)
  - Hybrid source: `titleblock` (30/30, mean confidence `0.85`)

Notes:
- Graph2D status is `model_unavailable` in this environment (torch not installed); HybridClassifier remains functional via rules/text.

