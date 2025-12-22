#!/usr/bin/env markdown
# Week 4 Report: Metrics Loop

## Summary
- Added DFM and cost estimation latency metrics and wired them into analysis flow.

## Changes
- Metrics: `dfm_analysis_latency_seconds`, `cost_estimation_latency_seconds`
- Wiring in `src/api/v1/analyze.py`
- Doc update: `docs/METRICS_INDEX.md`

## Tests
- Not run (requires service metrics scrape).

## Verification
- Hit `/api/v1/analyze` with DFM and cost enabled; verify `/metrics` includes:
  - `dfm_analysis_latency_seconds_bucket`
  - `cost_estimation_latency_seconds_bucket`
