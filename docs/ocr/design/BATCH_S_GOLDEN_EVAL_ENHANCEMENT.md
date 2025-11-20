# Batch S — Golden Evaluation Enhancement (Calibration Metrics)

Scope
- Extend golden evaluation to analyze confidence calibration (Brier, ECE, buckets) using per-item confidence.

Changes
- Updated `tests/ocr/golden/run_golden_evaluation.py`:
  - Collect per-dimension `confidence` from structured results.
  - Compute Brier overall and per confidence bucket.
  - Compute Expected Calibration Error (ECE) over bins: [0.0,0.6,0.8,0.9,1.0].
  - Output second report `reports/ocr_calibration.md` with bucket table.
  - JSON stdout now includes `calibration` section for CI parsing.

Rationale
- Raw average metrics (recall, edge_f1) insufficient; calibration quality informs fallback threshold tuning & MultiEvidenceCalibrator evolution.

Acceptance
- Script runs locally (requires Python env) producing both reports without exceptions.
- No change in existing unit test outcomes.

Future Work
- Integrate strict gate on Brier and ECE for Week2 (soft until stable).
- Add symbol-level calibration once symbol confidences are reliable.
- Introduce reliability plot artifact (PNG) generation (optional).

