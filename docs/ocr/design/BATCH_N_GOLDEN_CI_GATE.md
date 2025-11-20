# Batch N — Golden Evaluation CI Gate

Scope
- Integrate golden evaluation into CI with a soft gate initially; plan to flip to strict once metrics stabilize.

Design
- Script: `tests/ocr/golden/run_golden_evaluation.py` runs manager pre/post preprocessing, computes metrics, writes `reports/ocr_evaluation.md`.
- CI workflow `.github/workflows/ci.yml` calls the script; currently non-blocking via `|| echo`.
- Thresholds defined in `tests/ocr/golden/metadata.yaml` (Week1/Week2).

Next Step
- Flip to strict by removing `|| echo` and letting non-zero exit fail the job once Week1 thresholds are consistently met.

Acceptance
- CI artifacts include the evaluation report.
- Developers can run the script locally for comparison and to validate parsing changes.

