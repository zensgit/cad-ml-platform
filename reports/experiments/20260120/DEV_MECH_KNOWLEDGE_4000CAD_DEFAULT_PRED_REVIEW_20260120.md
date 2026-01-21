# DEV_MECH_KNOWLEDGE_4000CAD_DEFAULT_PRED_REVIEW_20260120

## Summary
Prepared a 20-sample review sheet from the default model batch predictions and
auto-filled verdicts using manifest labels as the reference.

## Steps
- Sampled 20 rows (seed: 20260120) from `reports/MECH_4000_DWG_DEFAULT_PRED_20260120.csv`.
- Created a review worksheet with verdict and corrected label fields.
- Auto-filled `review_verdict` / `corrected_label` based on `label_cn`.

## Results
- correct: 5
- incorrect: 15
- sample Top-1: 0.25

## Output
- `reports/MECH_4000_DWG_DEFAULT_PRED_REVIEW_20260120.csv`

## How to Review
- Fill `review_verdict` with one of: `correct`, `incorrect`, `uncertain`.
- If incorrect, add `corrected_label`.
- Optional: add clarifying `notes`.

## Next Step
- If you want true manual review, replace the auto-filled verdicts with human
  inspection and re-run the accuracy summary.
