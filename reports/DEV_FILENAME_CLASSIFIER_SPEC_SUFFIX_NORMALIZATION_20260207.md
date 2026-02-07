# DEV_FILENAME_CLASSIFIER_SPEC_SUFFIX_NORMALIZATION_20260207

## Goal

Eliminate remaining low-confidence DXF classifications caused by filename part names carrying trailing spec tokens (example: `拖车DN1500`), which prevented `HybridClassifier` from overriding placeholder/low-confidence rule outputs.

## Change Summary

- Updated `src/ml/filename_classifier.py`:
  - Added conservative part-name normalization that strips common trailing spec suffixes when they appear at the very end of the extracted name:
    - `DN####` (diameter nominal)
    - `PN##` (pressure rating)
    - `M##` / `M##x##` (thread spec)
  - Applied normalization at extraction time so downstream matching becomes exact where possible (example: `拖车DN1500` -> `拖车`).

## Verification

### Unit/Integration Tests

- Ran:
  - `.venv/bin/pytest -q tests/unit/test_filename_classifier.py tests/unit/test_hybrid_classifier.py`
    - Result: `41 passed`
  - `.venv/bin/pytest -q tests/integration/test_analyze_dxf_hybrid_override.py`
    - Result: `3 passed`

### Local Batch Evaluation (110 training DXF files)

- Ran (local TestClient batch):
  - `XDG_CACHE_HOME=/tmp/xdg-cache GRAPH2D_ENABLED=false GRAPH2D_FUSION_ENABLED=false FUSION_ANALYZER_ENABLED=false HYBRID_CLASSIFIER_ENABLED=true HYBRID_CLASSIFIER_AUTO_OVERRIDE=true .venv/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" --output-dir reports/experiments/20260207/batch_analyze_dxf_local_hybrid_v3 --seed 2207 --min-confidence 0.8`

- Results (from `reports/experiments/20260207/batch_analyze_dxf_local_hybrid_v3/summary.json`):
  - `total=110`, `success=110`, `error=0`
  - `low_confidence_count=0` (previous run `v2` had `2`)
  - Label distribution improvement:
    - `拖车: 2`
    - `complex_assembly: 0` (previous run `v2` had `2`)

### Artifact Hygiene

- Updated `scripts/batch_analyze_dxf_local.py` to:
  - Write a sanitized results CSV (`batch_results_sanitized.csv`) with only basenames in the `file` column.
  - Emit a local `.gitignore` in the output directory to keep raw CSVs (with absolute paths) untracked.

## Notes / Limits

- `torch` is not installed in this `.venv`, so V16 model loading falls back to rules; this verification focuses on filename-driven Hybrid override behavior.

