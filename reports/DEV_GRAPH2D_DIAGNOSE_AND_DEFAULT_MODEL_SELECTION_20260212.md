# DEV_GRAPH2D_DIAGNOSE_AND_DEFAULT_MODEL_SELECTION_20260212

## Goal
Diagnose why Graph2D was not producing useful part-name predictions on a local DXF training set, and choose a safer default `GRAPH2D_MODEL_PATH`.

Constraints:
- Manual DXF review is not feasible.
- Prefer deterministic, script-driven diagnostics.
- Keep Graph2D conservative: Hybrid is still filename-first; Graph2D is optional and gated.

## Dataset / Ground Truth (Weak)
- DXF directory: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`
- Files sampled: `110` (all files in the directory)
- Ground-truth mode: `--labels-from-filename`
  - Uses `FilenameClassifier` + `data/knowledge/label_synonyms_template.json` to extract/canonicalize labels.
  - Accept as true label only when `confidence >= 0.8`.

This is weak supervision (filename-derived labels), but it is good enough to detect label-space mismatch and obvious model collapse.

## How To Reproduce
```bash
python3 scripts/diagnose_graph2d_on_dxf_dir.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --labels-from-filename \
  --true-label-min-confidence 0.8 \
  --max-files 200 \
  --seed 42 \
  --model-path models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth \
  --output-dir reports/experiments/20260212/graph2d_diagnose/training_dxf_oda_titleblock_distill_20260210
```

## Results Summary
Metrics are from `summary.json` emitted by `scripts/diagnose_graph2d_on_dxf_dir.py`.

| run | model | labels | p50_conf | p90_conf | accuracy | top_pred |
|---|---|---:|---:|---:|---:|---|
| training_dxf_drawings_20260208 | `models/graph2d_training_drawings_20260208.pth` | 47 | 0.0344 | 0.0403 | 0.0727 | 阀体 (60) |
| training_dxf_latest_114 | `models/graph2d_latest.pth` | 114 | 0.0594 | 0.2966 | 0.0000 | 机械制图 (110) |
| training_dxf_oda_titleblock_distill_20260210 | `models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth` | 47 | 0.0406 | 0.0909 | 0.2727 | 轴向定位轴承 (36) |
| training_dxf_parts_filename_synonyms_20260126 | `models/graph2d_parts_filename_synonyms_20260126.pth` | 47 | 0.0276 | 0.0280 | 0.0364 | 拖车 (57) |
| training_dxf_parts_upsampled_20260122 | `models/graph2d_parts_upsampled_20260122.pth` | 33 | 0.3145 | 0.4177 | 0.0000 | 机械制图 (79) |
| training_dxf_training_20260123 | `models/graph2d_training_20260123.pth` | 47 | 0.0341 | 0.1141 | 0.0636 | 罐体部分 (45) |
| training_dxf_training_cleaned_20260123 | `models/graph2d_training_cleaned_20260123.pth` | 12 | 0.7457 | 0.9228 | 0.0273 | other (106) |

Key findings:
- The previous default (`graph2d_parts_upsampled_20260122.pth`) is not in the same label-space as the training DXF part labels (it behaves like a drawing-type classifier). It also has high confidence, which is risky when enabled.
- `graph2d_latest.pth` fully collapses to a single drawing-type label (`机械制图`).
- `graph2d_training_cleaned_20260123.pth` is a small-label (12-class) model that predicts `other` for almost everything. It can be useful for a narrow allowlist override workflow, but it is not suitable as a general default for part-name classification.
- The best-performing model in this diagnosis set is:
  - `models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth`
  - It has the same label-space size as the filename-derived label set (`47` distinct labels here) and achieves the highest top-1 accuracy (`~27%`).
  - Confidence is low (many-class softmax), but Hybrid already applies dynamic min-confidence gating based on `label_map_size`.

## Decision
Update the default `GRAPH2D_MODEL_PATH` fallback (used when env var is not set) to:
- `models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth`

This makes "turn on Graph2D" safer by default:
- avoids drawing-type label-space mismatch
- avoids the previous high-confidence wrong predictions

## Code Changes
Updated default model-path fallbacks in:
- `src/ml/vision_2d.py`
- `src/core/providers/classifier.py`
- `src/main.py`
- `src/api/health_utils.py`
- `scripts/calibrate_graph2d_temperature.py`
- `scripts/diagnose_graph2d_on_dxf_dir.py`
- `docs/HEALTH_ENDPOINT_CONFIG.md`

## Artifacts
Summary artifacts (per-run):
- `reports/experiments/20260212/graph2d_diagnose/*/summary.json`

Note: per-file predictions are written to `predictions.csv` and are intentionally ignored by `.gitignore` in each run directory.

## Validation
- `make validate-core-fast` (passed)
