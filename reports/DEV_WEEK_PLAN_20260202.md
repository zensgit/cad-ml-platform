# DEV_WEEK_PLAN_20260202

## Goal
Stabilize tolerance knowledge, benchmark suite, and dependency readiness; enable ISO 286 hole deviation overrides and keep performance validation reproducible.

## Day 1 – Baseline & Environment
- Verify existing tolerance knowledge functions and identify non‑H fit gaps.
- Validate benchmark suite runs in a clean environment.
- Add missing runtime dependency (Pillow) to requirements for analyze imports.

## Day 2 – ISO 286 Overrides (Data Plumbing)
- Add `HOLE_DEVIATIONS_PATH` override support with JSON schema template.
- Enable per-symbol deviation overrides (EI) when JSON is provided.

## Day 3 – Fit Coverage Expansion
- Add representative shaft‑basis fit codes for keyway scenarios.
- Extend tests for non‑H hole fits and override loading behavior.

## Day 4 – Benchmark Stability
- Stabilize benchmark HTTP clients (async transport).
- Add warm‑up to avoid cold‑start latency skew.

## Day 5 – Validation Pass
- Run unit tests for GD&T/tolerance.
- Run full benchmark suite and capture warnings.

## Day 6 – Documentation
- Update design and validation reports with override path and results.
- Record benchmark execution in verification log.

## Day 7 – Buffer & Review
- Leave a buffer for additional ISO table data ingestion if provided.
- Review and confirm no regressions in fits computations.

## Status
- All steps executed in this iteration.
