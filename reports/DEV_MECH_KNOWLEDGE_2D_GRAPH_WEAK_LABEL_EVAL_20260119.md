# DEV_MECH_KNOWLEDGE_2D_GRAPH_WEAK_LABEL_EVAL_20260119

## Summary
Computed a weak-label baseline for the graph2d checkpoint using 20 DXF samples
from the manifest. Labels come from filename-derived tags (not manual ground truth).

## Method
- Sampled 20 entries from `reports/MECH_DWG_LABEL_MANIFEST_20260119.csv` (seed=13).
- Loaded `models/graph2d_latest.pth` and ran inference on matching DXF files.
- Compared predictions to manifest labels (weak labels).

## Results
- Sample size: 20
- Top-1 accuracy: 0.15
- Top-3 accuracy: 0.25

## Notes
- This is a weak-label baseline; manual review is still required for true accuracy.
- Errors are expected because label quality is based on filenames and the model
  is trained on weak labels.
