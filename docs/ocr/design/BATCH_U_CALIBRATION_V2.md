# Batch U — MultiEvidenceCalibrator v2

Scope
- Extend calibration to incorporate additional evidence and adaptive weighting.

Evidence Inputs
- raw_confidence: provider base score.
- completeness: parsed key token coverage ratio.
- item_mean: mean per-item (dimension + symbol) confidence.
- fallback_recent: recent fallback ratio (penalized).
- parse_error_rate: recent parse error ratio (penalized).

Weights (initial)
- raw 0.5, completeness 0.25, item_mean 0.15, fallback_recent 0.05, parse_error 0.05.

Adaptive Logic
- If Brier > 0.3: shift weights toward completeness & item_mean, reduce raw.
- If Brier < 0.15: reward raw, trim completeness/item_mean slightly.

Integration
- Manager now computes `item_mean`; placeholder zeros for recent ratios (future: sliding window metrics).
- Fallback paths also apply v2 calibration with available evidence.

Testing
- Added `tests/ocr/test_calibrator_v2.py` covering evidence blend, missing raw scenario, adaptive reweight behavior.

Future Work
- Persist recent fallback & parse error counters (Redis or in-memory ring buffer).
- Feed calibration Brier/ECE from golden evaluation to adaptive loop.
- Export per-evidence contribution metrics for observability.

Risks
- Over-adjustment of weights without stable signal; keep adaptation deltas small.

