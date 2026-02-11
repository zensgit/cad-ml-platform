# DEV_MAKE_GRAPH2D_REVIEW_SUMMARY_20260211

## Summary
- Added a stable Makefile entrypoint for Graph2D soft-override review summarization.
- This reduces manual script invocation differences across environments.

## Changes
- Updated `Makefile`
  - Added `.PHONY` entry: `graph2d-review-summary`
  - Added target variables:
    - `GRAPH2D_REVIEW_TEMPLATE`
    - `GRAPH2D_REVIEW_OUT_DIR`
  - Added target: `make graph2d-review-summary`

## Validation
- `make graph2d-review-summary GRAPH2D_REVIEW_OUT_DIR=/tmp/graph2d_review_summary_20260211`
  - Summary output: `/tmp/graph2d_review_summary_20260211/soft_override_review_summary.csv`
  - Correct-label output: `/tmp/graph2d_review_summary_20260211/soft_override_correct_label_counts.csv`
  - Summary sample:
    - `total=15`
    - `reviewed=15`
    - `agree_with_graph2d=0`
    - `disagree_with_graph2d=15`
    - `agree_rate=0.0000`

