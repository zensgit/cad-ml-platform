# Batch R — Golden Dataset Expansion Phase 2

Scope
- Add medium/hard/edge samples to diversify evaluation for Week2+ metrics.

Additions
- `sample_004` (medium): mixed dimensions + roughness, single tolerance.
- `sample_005` (hard): dual tolerance + thread missing pitch (tests parser robustness).
- `sample_006` (edge): extreme tolerances and missing symbols to test fallback completeness.
- `sample_007` (edge): symbol-only case (no dimensions) validates recall logic when token hints exist.
- `sample_008` (hard): multiple small radius values with tight tolerances.

Metadata Updates
- Category counts adjusted; Week3 thresholds added.

Acceptance
- Golden evaluation script runs without errors.
- Unit tests remain green.
- Threshold gating remains soft until stabilization.

