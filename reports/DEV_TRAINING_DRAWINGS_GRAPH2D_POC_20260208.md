# DEV_TRAINING_DRAWINGS_GRAPH2D_POC_20260208

## Goal
Run a **fully automated** (no manual drawing review) Graph2D training + evaluation loop on the local training DXF set, using weak labels derived from filenames.

This is primarily a **pipeline validation** (manifest -> train -> eval -> artifacts), and provides a baseline for future iterations (more data, better sampling, multimodal fusion, distillation).

## Dataset
- DXF directory: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`
- Files scanned: `110`

## Label Manifest
Generated with `FilenameClassifier` + synonym standardization.

Command:
```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/build_dxf_label_manifest.py \
  --input-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --label-mode filename \
  --output-csv reports/experiments/20260208/TRAINING_DXF_LABEL_MANIFEST_20260208.csv
```

Observed summary:
- `110/110` rows matched a standardized label (`label_status=matched`, `match_type=exact`)
- Unique labels: `47`

Artifact:
- `reports/experiments/20260208/TRAINING_DXF_LABEL_MANIFEST_20260208.csv`

### Filename Extraction Fix
One filename contained a trailing vendor suffix (`...v2-yuantus.dxf`) which previously polluted the extracted label as `yuantus`.

Change:
- `src/ml/filename_classifier.py`: prefer Chinese substrings when ASCII suffixes are present (keeps labels stable and synonym-matchable).

Validation:
- `pytest -q tests/unit/test_filename_classifier.py::test_extract_part_name_patterns` (passed)

## Training
Command:
```bash
PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/tmp/xdg-cache python3 scripts/train_2d_graph.py \
  --manifest reports/experiments/20260208/TRAINING_DXF_LABEL_MANIFEST_20260208.csv \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --epochs 20 \
  --batch-size 4 \
  --hidden-dim 128 \
  --lr 0.001 \
  --node-dim 19 \
  --model edge_sage \
  --edge-dim 7 \
  --loss focal \
  --class-weighting sqrt \
  --sampler balanced \
  --save-best \
  --early-stop-patience 5 \
  --output models/graph2d_training_drawings_20260208.pth
```

Key output:
- Best `val_acc=0.085` (early stopped at epoch `8`)

Artifact:
- `models/graph2d_training_drawings_20260208.pth`
  - Note: `models/*.pth` is git-ignored in this repo, so the checkpoint is a local artifact unless you explicitly add an exception.

## Evaluation
Command:
```bash
PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/tmp/xdg-cache python3 scripts/eval_2d_graph.py \
  --manifest reports/experiments/20260208/TRAINING_DXF_LABEL_MANIFEST_20260208.csv \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --checkpoint models/graph2d_training_drawings_20260208.pth \
  --output-metrics reports/experiments/20260208/TRAINING_DXF_GRAPH2D_VAL_METRICS_20260208.csv \
  --output-errors reports/experiments/20260208/TRAINING_DXF_GRAPH2D_VAL_ERRORS_20260208.csv
```

Observed summary:
- `Validation samples=47`
- `acc=0.085`, `top2=0.085`
- `macro_f1=0.039`, `weighted_f1=0.039`

Artifacts:
- `reports/experiments/20260208/TRAINING_DXF_GRAPH2D_VAL_METRICS_20260208.csv`
- `reports/experiments/20260208/TRAINING_DXF_GRAPH2D_VAL_ERRORS_20260208.csv`

## Interpretation / Next Steps
This result confirms the **pipeline works**, but also shows a geometry-only Graph2D model cannot learn 47 fine-grained part labels from only 110 drawings.

Recommended next iteration (highest ROI):
1. **Reduce label space** (coarse taxonomy) for Graph2D (e.g. 10-20 buckets), keep fine label from Filename/TitleBlock rules.
2. **Expand training data** (more DXF files per label). Target: 20-50 samples per label minimum for a meaningful geometry model.
3. **Multimodal fusion** (already implemented in `HybridClassifier`):
   - Graph2D: geometry embedding
   - FilenameClassifier/TitleBlock: strong weak-supervision
4. Optional: **Distillation** (teacher=Hybrid/Filename) to improve behavior when filenames are missing/obfuscated.
