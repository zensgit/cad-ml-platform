# Batch P — Per-item Confidence Propagation

Scope
- Carry OCR line confidence scores (where available) into structured outputs for each dimension and symbol.

Design
- Models: add `confidence: float|None` to `DimensionInfo` and `SymbolInfo`.
- BBox mapper: accept optional `score` in OCR lines and set per-item confidence when assigning bboxes.
- Providers: when converting PaddleOCR output to `ocr_lines`, include `score` field.
- DeepSeek alignment path: include `score` similarly when using Paddle alignment.

Rationale
- Enables finer-grained calibration and reliability metrics (e.g., Brier by bucket), and downstream gating.

Acceptance
- Tests remain green; golden evaluation script can compute Brier using per-item confidence if present.

Notes
- If provider doesn’t expose line score, we leave per-item confidence as None and fallback to defaults in evaluation.

