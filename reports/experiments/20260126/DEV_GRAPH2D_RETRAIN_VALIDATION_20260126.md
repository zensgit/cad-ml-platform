# DEV_GRAPH2D_RETRAIN_VALIDATION_20260126

## Validation summary
- Manifest coverage: 109/110 matched labels (99.09%), 47 unique labels.
- Training: 5 epochs, focal loss + balanced sampling; checkpoint saved.
- Evaluation: `acc=0.043`, `top2=0.064`, `macro_f1=0.003`, `weighted_f1=0.003` on 47 validation samples.
- Batch review vs filename: 4 agree / 105 disagree / 1 unknown (agreement rate 3.64%).

## Evidence
### Manifest
- `reports/experiments/20260126/graph2d_retrain_manifest_20260126/dxf_manifest_summary_20260126.json`
- `reports/experiments/20260126/graph2d_retrain_manifest_20260126/dxf_manifest_label_counts_20260126.csv`

### Training
- `models/graph2d_parts_filename_synonyms_20260126.pth`

### Evaluation
- `reports/experiments/20260126/graph2d_retrain_eval_metrics_20260126.csv`
- `reports/experiments/20260126/graph2d_retrain_eval_errors_20260126.csv`

### Batch review (new model)
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_retrained_20260126/soft_override_review_summary_20260126.csv`
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_retrained_20260126/soft_override_conflicts_20260126.csv`
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_retrained_20260126/filename_coverage_summary_20260126.csv`

## Environment note
- MPS backend failed for class-weighted focal loss (`cross_entropy_loss` placeholder storage). Training was rerun on CPU.

## Conclusion
- With only 110 samples / 47 labels, the retrained Graph2D remains underfit. Agreement with filename-derived labels is still low; a larger labeled dataset or distillation from filename/title-block signals is needed before enabling Graph2D for part-type decisions.
