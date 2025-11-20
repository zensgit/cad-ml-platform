# Batch L — BBox Heuristics

Status: completed

Scope
- Improve bbox assignment for parsed dimensions/symbols when exact text or value substring match fails.

Approach
- Keep existing matching order:
  1) Exact/substring match on raw token
  2) Substring match on numeric value
  3) Heuristic scoring fallback
- Heuristic score combines:
  - String similarity (difflib) between compact needle (raw or "<type> <value>") and OCR line text (60%)
  - Numeric proximity of numbers in line to the target value (30%)
  - Type hint bonus when line includes glyph/token (Φ/R/M) aligned to type (10%)
- Apply when score ≥ 0.60 for dimensions; symbols use a simpler similarity fallback with threshold 0.65.

Files
- Updated: `src/core/ocr/parsing/bbox_mapper.py`

Risks & Mitigations
- Possible false positives in dense text blocks — thresholds tuned conservatively and only used when direct matches fail.
- Keep logic deterministic and inexpensive (no heavy NLP deps).

Acceptance
- Existing tests remain green.
- Add a focused test to exercise heuristic path.

Next
- Optional: add debug logging for top-N candidate lines and scores when near threshold.
- Consider integrating bbox confidence to downstream calibration.

