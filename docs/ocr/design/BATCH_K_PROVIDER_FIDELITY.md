# Batch K — Provider Fidelity (Paddle)

Status: completed

Scope
- Improve Paddle provider robustness and configurability without changing APIs.

Changes
- `src/core/ocr/providers/paddle.py`
  - Accepts arbitrary `**paddle_kwargs` to tune PaddleOCR (e.g., det/rec models, langs, rec_algorithm).
  - Defensive parsing of OCR outputs across PaddleOCR versions: supports tuple and dict-like forms.
  - Maintains CPU-first defaults: `lang='ch', use_angle_cls=True, use_gpu=False`.
  - Safe fallbacks on exceptions; regex parser can still extract essentials when OCR fails.

Rationale
- Different PaddleOCR builds return slightly different structures; we must parse flexibly.
- Fidelity tuning via kwargs allows future controlled experiments without code changes.

Acceptance
- All unit tests pass.
- Provider can initialize with defaults and with overridden kwargs.
- No API changes required.

Next
- Heuristic bbox mapping improvements (string similarity, numeric proximity).
- Provider-level structured logs per stage (infer/parse/align).

