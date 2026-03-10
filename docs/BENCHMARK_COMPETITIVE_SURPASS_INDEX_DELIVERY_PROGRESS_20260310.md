# Benchmark Competitive Surpass Index Delivery Progress

## Scope

This stack lifts `competitive_surpass_index` from a standalone benchmark
exporter to a first-class release surface.

Implemented layers:

1. standalone exporter
2. artifact bundle surface
3. companion summary surface
4. CI / job summary surface
5. PR comment / signal-light surface
6. release decision surface
7. release runbook surface
8. release CI / PR comment surface

## Delivered Branch Stack

- `#293` `feat: add benchmark competitive surpass index`
- `#294` `feat: wire benchmark competitive surpass index into ci`
- `#295` `feat: add benchmark competitive surpass index pr comment`
- `#296` `feat: add benchmark competitive surpass index release surfaces`
- `#297` `feat: wire benchmark competitive surpass release surfaces into ci`

## Output Contract

Core fields:

- `status`
- `score`
- `ready_pillars`
- `partial_pillars`
- `blocked_pillars`
- `primary_gaps`
- `recommendations`

Release surfaces additionally expose:

- `competitive_surpass_index_status`
- `competitive_surpass_primary_gaps`
- `competitive_surpass_recommendations`

## Validation Docs

- `docs/BENCHMARK_COMPETITIVE_SURPASS_INDEX_VALIDATION_20260310.md`
- `docs/BENCHMARK_COMPETITIVE_SURPASS_INDEX_CI_VALIDATION_20260310.md`
- `docs/BENCHMARK_COMPETITIVE_SURPASS_INDEX_PR_COMMENT_VALIDATION_20260310.md`
- `docs/BENCHMARK_COMPETITIVE_SURPASS_INDEX_RELEASE_SURFACES_VALIDATION_20260310.md`
- `docs/BENCHMARK_COMPETITIVE_SURPASS_INDEX_RELEASE_SURFACES_CI_VALIDATION_20260310.md`

## Remaining Work

- merge the stacked PR chain into `main`
- rebase any downstream benchmark control-plane work on top of the release-CI layer
- if needed, split release-surface signal lights into a dedicated follow-up PR
  once this stack is fully absorbed
