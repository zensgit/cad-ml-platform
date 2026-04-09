# Benchmark Release Decision Delivery Progress

## Purpose

`benchmark release decision` is the next benchmark layer above:

- scorecard
- operational summary
- artifact bundle
- companion summary

It answers a different question:

- not just "what signals exist?"
- but "is this run ready, review-required, or blocked?"

## Delivered

### Standalone Export

Implemented:

- `scripts/export_benchmark_release_decision.py`
- JSON + Markdown outputs
- stable fields:
  - `release_status`
  - `automation_ready`
  - `primary_signal_source`
  - `blocking_signals`
  - `review_signals`
  - `component_statuses`

### CI Wiring

Implemented:

- optional workflow inputs
- optional build step
- artifact upload
- job summary lines

### Bundle Integration

Implemented:

- benchmark artifact bundle can ingest release decision JSON
- bundle summary can prefer release decision state
- CI bundle path can pass release decision from workflow inputs or step outputs

### PR Review Surface

Implemented:

- release decision is visible in benchmark PR comment payloads
- duplicate artifact-bundle and companion rows were removed from the comment

## Why This Matters for Competitive Surpass

This layer moves the platform closer to operator-facing judgment rather than
raw extraction or benchmarking only.

That is important for surpassing OCR-first or extraction-first products because
the release decision layer:

- translates benchmark state into an operational recommendation
- keeps human review in the loop when automation is unsafe
- creates a concrete bridge from evaluation artifacts to release governance

## Current State

The benchmark stack now has these layers:

1. `scorecard`
2. `feedback flywheel benchmark`
3. `operational summary`
4. `artifact bundle`
5. `companion summary`
6. `release decision`

## Remaining Next Steps

1. Merge and stabilize the release-decision PR stack on `main`
2. Add a release-decision-specific artifact/PR comment summary rollup if needed
3. Connect real-data validation reports to release decision generation
4. Surface standards/tolerance/GD&T blockers directly into release decision
