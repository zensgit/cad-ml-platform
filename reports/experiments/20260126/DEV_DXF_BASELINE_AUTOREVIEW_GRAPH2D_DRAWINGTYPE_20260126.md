# DEV_DXF_BASELINE_AUTOREVIEW_GRAPH2D_DRAWINGTYPE_20260126

## Goal
Exclude Graph2D drawing-type labels from part-name fusion and re-run the automated DXF baseline review with Graph2D enabled.

## Scope
- Dataset: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`
- Requested sample size: 300
- Available DXF files: 110 (sample size capped at 110)
- Runtime: `.venv-graph/bin/python`
- Drawing-type labels: `零件图, 机械制图, 装配图, 练习零件图, 原理图, 模板`

## Work performed
1. Update Graph2D fusion logic to ignore drawing-type labels for part-name decisions
2. Tag Graph2D output with `is_drawing_type`
3. Mark drawing-type outputs in API and soft-override eligibility
4. Re-run baseline batch analysis (Graph2D enabled)
5. Auto-review via filename + synonym matching
6. Generate review summaries, conflicts, and coverage CSVs

## Commands
```bash
TITLEBLOCK_OVERRIDE_ENABLED=false \
GRAPH2D_DRAWING_TYPE_LABELS="零件图,机械制图,装配图,练习零件图,原理图,模板" \
  .venv-graph/bin/python scripts/batch_analyze_dxf_local.py \
    --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
    --max-files 300 \
    --seed 20260126 \
    --output-dir "reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_drawingtype_20260126"

.venv-graph/bin/python scripts/review_soft_override_batch.py \
  --input "reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_drawingtype_20260126/batch_results.csv" \
  --output "reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_drawingtype_20260126/soft_override_reviewed_20260126.csv" \
  --reviewer auto

.venv-graph/bin/python scripts/summarize_soft_override_review.py \
  --review-template "reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_drawingtype_20260126/soft_override_reviewed_20260126.csv" \
  --summary-out "reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_drawingtype_20260126/soft_override_review_summary_20260126.csv" \
  --correct-labels-out "reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_drawingtype_20260126/soft_override_correct_label_counts_20260126.csv"
```

## Code changes
- `src/ml/hybrid_classifier.py`
  - Added drawing-type allowlist and ignore graph2d labels in fusion decisions when marked.
  - Tagged Graph2D predictions with `is_drawing_type` and added decision path markers.
- `src/api/v1/analyze.py`
  - Added `_graph2d_is_drawing_type` helper and excluded drawing-type labels from fusable/override logic.
  - Graph2D predictions now include `is_drawing_type`.
- `scripts/batch_analyze_dxf_local.py`
  - Added `graph2d_is_drawing_type` column to batch output.

## Outputs
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_drawingtype_20260126/batch_results.csv`
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_drawingtype_20260126/batch_low_confidence.csv`
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_drawingtype_20260126/summary.json`
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_drawingtype_20260126/label_distribution.csv`
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_drawingtype_20260126/soft_override_reviewed_20260126.csv`
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_drawingtype_20260126/soft_override_review_summary_20260126.csv`
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_drawingtype_20260126/soft_override_correct_label_counts_20260126.csv`
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_drawingtype_20260126/soft_override_conflicts_20260126.csv`
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_drawingtype_20260126/filename_coverage_summary_20260126.csv`

## Notes
- The dataset contains 110 DXFs, so the 300-sample target was capped.
- Graph2D predictions are now tagged as drawing-type; these signals no longer participate in part-name fusion or override eligibility.
