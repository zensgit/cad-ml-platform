# DEV_GRAPH2D_DIAGNOSE_TRAINING_DXF_ODA_20260209

## Goal
Diagnose the current Graph2D classifier behavior on a real training DXF directory, without manual drawing review.

We use weak supervision by deriving `true_label` from `FilenameClassifier` (with canonicalization via the synonyms table), then compare it to Graph2D predictions.

## Run
Dataset:
- Input: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123` (110 DXF)

Command (baseline checkpoint used previously by Graph2D module defaults):
```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
XDG_CACHE_HOME=/tmp/xdg-cache \
  .venv-graph/bin/python scripts/diagnose_graph2d_on_dxf_dir.py \
    --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" \
    --model-path models/graph2d_parts_upsampled_20260122.pth \
    --max-files 110 \
    --seed 22 \
    --output-dir reports/experiments/20260209/graph2d_diagnose_parts_upsampled \
    --labels-from-filename \
    --true-label-min-confidence 0.8
```

Command (checkpoint trained on drawing-part labels):
```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
XDG_CACHE_HOME=/tmp/xdg-cache \
  .venv-graph/bin/python scripts/diagnose_graph2d_on_dxf_dir.py \
    --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" \
    --model-path models/graph2d_training_drawings_20260208.pth \
    --max-files 110 \
    --seed 22 \
    --output-dir reports/experiments/20260209/graph2d_diagnose_training_drawings \
    --labels-from-filename \
    --true-label-min-confidence 0.8
```

Outputs:
- `reports/experiments/20260209/graph2d_diagnose_parts_upsampled/summary.json`
- `reports/experiments/20260209/graph2d_diagnose_training_drawings/summary.json`
- Per-file `predictions.csv` is written but **gitignored** in each output dir (contains file names).

## Results Summary
### 1) `graph2d_parts_upsampled_20260122.pth` (label_map=33)
- Graph2D predicted labels are almost entirely **drawing-type** classes:
  - top preds: `机械制图=79`, `零件图=16`, `装配图=13`, `模板=2`
- Weak-supervised accuracy vs filename labels: `0.0`
- Confidence:
  - `p50=0.3145`, `p90=0.4177`

Interpretation:
- This checkpoint is not suitable for fine part-label classification for this dataset; it behaves like a drawing-type classifier.

### 2) `graph2d_training_drawings_20260208.pth` (label_map=47)
- Graph2D prediction collapsed to a few labels:
  - top preds: `阀体=60`, `液压开盖组件=43`, `旋转组件=5`, `手轮组件=2`
- Weak-supervised accuracy vs filename labels: `0.0727`
- Confidence:
  - `p50=0.0344`, `p90=0.0403`

Interpretation:
- Even with a label set aligned to the dataset, confidence is near-uniformly low and predictions collapse to a few classes, suggesting the geometry-only GNN is not learning stable separations for this label space (or the checkpoint is underfit / mismatched to current graph extraction).

## Conclusion / Next Steps
- Graph2D should be treated as a **low-trust auxiliary signal** for now, and should not override high-confidence signals from:
  - `FilenameClassifier` and/or
  - `TitleBlockExtractor`
- If we need a filename/titleblock-free classifier, the next engineering step is to:
  - retrain Graph2D with a clearer task definition (coarse family vs fine label),
  - apply class balancing, and
  - consider knowledge distillation (filename/titleblock teacher -> geometry student) for robustness when filenames are unavailable.

