# Benchmark Release Scorecard Operator Adoption Progress

## Scope

Track the stacked delivery of scorecard-level and operational-level
operator-adoption visibility into the benchmark release surfaces.

Stack:

1. `#281` release decision / release runbook exporter surfaces
2. `#282` workflow outputs and job summary wiring
3. `#283` PR comment and signal-light wiring

## What This Stack Adds

Release decision now exposes:

- `scorecard_operator_adoption`
- `operational_operator_adoption`
- scorecard operator outcome drift summary
- operational operator outcome drift summary

Release runbook now exposes the same two layers and their outcome-drift
summaries.

Workflow / review surfaces now expose:

- job summary lines for release decision scorecard / operational operator adoption
- job summary lines for release runbook scorecard / operational operator adoption
- PR comment rows for both release surfaces
- signal lights for both release surfaces

## Design Intent

The release surfaces already carried standalone `operator_adoption` and
operator-drift status. This stack separates three layers that were previously
collapsed together:

- standalone benchmark operator adoption
- scorecard-derived operator adoption
- operational-summary-derived operator adoption

That separation matters because release review needs to tell apart:

- whether adoption is broadly ready at benchmark level
- whether operational governance is healthy
- whether the release-facing decision surfaces are still blocked or drifting

## Validation

Associated validation documents:

- `docs/BENCHMARK_RELEASE_SCORECARD_OPERATOR_ADOPTION_VALIDATION_20260309.md`
- `docs/BENCHMARK_RELEASE_SCORECARD_OPERATOR_ADOPTION_CI_VALIDATION_20260309.md`
- `docs/BENCHMARK_RELEASE_SCORECARD_OPERATOR_ADOPTION_PR_COMMENT_VALIDATION_20260309.md`

## Status

Current delivery state in this stack:

- exporter surfaces: implemented
- CI/job summary wiring: implemented
- PR comment/signal lights: implemented
- stacked PR chain: `#281 -> #282 -> #283`
