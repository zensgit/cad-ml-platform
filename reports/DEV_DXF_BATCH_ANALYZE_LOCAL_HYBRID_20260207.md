# DEV_DXF_BATCH_ANALYZE_LOCAL_HYBRID_20260207

## Goal
Run an end-to-end local batch evaluation of `/api/v1/analyze` over a DXF directory to validate Hybrid override behavior on real filenames, without requiring manual DXF rendering.

## Dataset
- Source dir: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`
- Files found: `110` DXF

## Commands
Baseline (before drawing-type override fix):
```bash
GRAPH2D_ENABLED=false GRAPH2D_FUSION_ENABLED=false FUSION_ANALYZER_ENABLED=false \
HYBRID_CLASSIFIER_ENABLED=true HYBRID_CLASSIFIER_AUTO_OVERRIDE=true \
.venv/bin/python scripts/batch_analyze_dxf_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --output-dir "reports/experiments/20260207/batch_analyze_dxf_local_hybrid" \
  --max-files 200 --seed 2207 --min-confidence 0.8
```

After fix (drawing-type labels no longer block Hybrid override):
```bash
GRAPH2D_ENABLED=false GRAPH2D_FUSION_ENABLED=false FUSION_ANALYZER_ENABLED=false \
HYBRID_CLASSIFIER_ENABLED=true HYBRID_CLASSIFIER_AUTO_OVERRIDE=true \
.venv/bin/python scripts/batch_analyze_dxf_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --output-dir "reports/experiments/20260207/batch_analyze_dxf_local_hybrid_v2" \
  --max-files 200 --seed 2207 --min-confidence 0.8
```

## Results Summary
| Metric | Baseline | After Fix |
|---|---:|---:|
| Total | 110 | 110 |
| Success | 110 | 110 |
| Errors | 0 | 0 |
| `机械制图` as `part_type` | 14 | 0 |
| Low-confidence (<= 0.8) | 2 | 2 |

### Remaining Low-Confidence Cases
The remaining 2 samples are `BTJ01231201522-00拖车DN1500v1/v2.dxf`, where filename matching is only `partial_low` (0.5) and `HybridClassifier` confidence stays below override threshold.

## Artifacts
Baseline dir:
- `reports/experiments/20260207/batch_analyze_dxf_local_hybrid/summary.json`
- `reports/experiments/20260207/batch_analyze_dxf_local_hybrid/label_distribution.csv`
- `reports/experiments/20260207/batch_analyze_dxf_local_hybrid/batch_results_sanitized.csv`
- `reports/experiments/20260207/batch_analyze_dxf_local_hybrid/batch_low_confidence_sanitized.csv`

After-fix dir:
- `reports/experiments/20260207/batch_analyze_dxf_local_hybrid_v2/summary.json`
- `reports/experiments/20260207/batch_analyze_dxf_local_hybrid_v2/label_distribution.csv`
- `reports/experiments/20260207/batch_analyze_dxf_local_hybrid_v2/batch_results_sanitized.csv`
- `reports/experiments/20260207/batch_analyze_dxf_local_hybrid_v2/batch_low_confidence_sanitized.csv`

Notes:
- The raw CSVs containing absolute local paths are present but ignored via per-folder `.gitignore`.
