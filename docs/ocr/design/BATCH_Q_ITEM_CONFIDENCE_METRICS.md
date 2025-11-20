# Batch Q — Per-item Confidence Metrics

Scope
- Expose histogram metrics for dimension/symbol confidence to support reliability analysis and future calibration refinement.

Design
- Metric: `ocr_item_confidence_distribution{provider,item_type}` with same bucket strategy as raw provider confidence.
- Emitted in `OcrManager` after successful extraction and any fallback application.
- Uses per-item `.confidence` from models populated during bbox assignment.

Rationale
- Enables Brier/Calibration assessment at granular level and monitoring drift over time (e.g., sudden drop in symbol confidence).

Acceptance
- Metric registered in `src/utils/metrics.py`.
- Manager emits observations for each item with non-null confidence.
- All tests remain green.

