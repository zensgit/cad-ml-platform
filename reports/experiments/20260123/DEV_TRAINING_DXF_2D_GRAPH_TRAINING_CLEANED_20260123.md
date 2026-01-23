# DEV_TRAINING_DXF_2D_GRAPH_TRAINING_CLEANED_20260123

## Summary
- Trained a 2D DXF graph classifier on the cleaned manifest with merged low-frequency labels.
- Downweighted the `other` class to reduce dominance.

## Output
- Checkpoint: `models/graph2d_training_cleaned_20260123.pth`
- Manifest: `reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_CLEANED_20260123.csv`

## Notes
- 15-epoch run reached validation accuracy in the 0.59–0.68 range (best observed 0.682).
