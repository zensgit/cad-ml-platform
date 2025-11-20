# Batch O — Golden Dataset Expansion (v1.0)

Scope
- Add more annotated samples across categories to stabilize evaluation and prepare for stricter CI gates.

Design
- New samples under `tests/ocr/golden/samples/` with `annotation.json` only (image bytes are stubbed in evaluator for MVP).
- Cover cases:
  - Dual tolerance binding (e.g., Φ20 +0.02 -0.01)
  - Thread pitch variants (M10×1.5)
  - Surface roughness symbol (Ra3.2)
  - Simple bbox references for IoU-based edge-F1

Changes
- Updated `tests/ocr/golden/metadata.yaml` category counts.
- Added three samples: `sample_002`, `sample_003`, `sample_004`.

Acceptance
- Golden evaluation script runs and produces a report over all samples.
- No unit test regressions.

