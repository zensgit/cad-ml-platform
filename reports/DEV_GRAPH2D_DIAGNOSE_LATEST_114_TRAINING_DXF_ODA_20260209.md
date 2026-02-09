# DEV_GRAPH2D_DIAGNOSE_LATEST_114_TRAINING_DXF_ODA_20260209

## Goal
Evaluate the `models/graph2d_latest.pth` checkpoint (label_map=114) on training DXFs using weak supervision from `FilenameClassifier` labels (synonym canonicalization), to understand whether “latest” provides useful fine-label geometry classification.

## Run
Command:
```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
XDG_CACHE_HOME=/tmp/xdg-cache \
  .venv-graph/bin/python scripts/diagnose_graph2d_on_dxf_dir.py \
    --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" \
    --model-path models/graph2d_latest.pth \
    --max-files 110 \
    --seed 22 \
    --output-dir reports/experiments/20260209/graph2d_diagnose_latest_114 \
    --labels-from-filename \
    --true-label-min-confidence 0.8
```

Outputs:
- `reports/experiments/20260209/graph2d_diagnose_latest_114/summary.json`
- Per-file `predictions.csv` is written but gitignored.

## Results (110 DXF)
From `summary.json`:
- `label_map_size`: `114`
- `top_pred_labels`: `机械制图=110` (single-label collapse)
- Weak-supervised accuracy vs filename labels: `0.0`
- Confidence distribution:
  - `p50=0.0594`, `p90=0.2966`

## Conclusion
`graph2d_latest.pth` is not suitable for fine-label part classification on this dataset. It behaves like a drawing-type detector and collapses to `机械制图`.

