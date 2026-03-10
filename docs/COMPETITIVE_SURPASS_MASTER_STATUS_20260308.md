# Competitive Surpass Master Status

## Goal

Move the platform beyond a pure CAD/OCR extraction service into an
engineering-semantic system with:

- stable coarse/fine classification contracts
- hybrid AI decisioning with explainability
- review and feedback flywheel governance
- benchmark scorecards and operational summaries
- artifact bundles and CI-visible delivery signals
- Qdrant-native retrieval, migration, and observability surfaces

## Delivered Areas

### 1. Stable AI Output Contracts

- `analyze`, `batch-classify`, similarity, compare, vector search, and provider
  bridges now expose coarse/fine label semantics
- final outputs preserve decision source and review-relevant metadata
- Graph2D has been repositioned as a weak signal under hybrid governance

### 2. Explainability and Review Governance

- assistant evidence output is structured and contract-tested
- active-learning review queue supports prioritization and export
- feedback, retraining, and metric-training contracts have been normalized
- OCR review guidance now emits actionable gap, priority, and automation hints

### 3. Benchmark Delivery Stack

- benchmark scorecard
- benchmark engineering signals
- benchmark knowledge readiness
- feedback flywheel benchmark
- operational summary
- artifact bundle
- companion summary
- release decision
- release runbook
- operator adoption

These layers now exist both as standalone exporters and as workflow-driven CI
surfaces.

### 4. CI / PR Observability

- `evaluation-report.yml` can build and upload benchmark engineering, bundle,
  knowledge-readiness, bundle, companion, release-decision, runbook, and
  operator-adoption artifacts
- PR comments and job summaries now surface bundle, companion, release, and
  engineering signals instead of only raw benchmark scorecard output
- PR comments and job summaries now also surface `knowledge=...` state across
  scorecard, bundle, companion, release decision, and release runbook layers
- companion summary is connected to bundle composition and engineering signals,
  so downstream reviewers see one compact operational picture
- release decision and release runbook now consume operator-adoption input in
  both standalone exporters and CI workflow surfaces
- release decision and release runbook now also expose scorecard-level and
  operational-level operator-adoption state, so reviewers can distinguish
  top-line adoption readiness from release-surface operator guidance
- release decision and release runbook also consume knowledge readiness, so
  release gating can reflect standards/tolerance/GD&T baseline completeness
- PR comments now expose release-surface `operator_adoption=...` state directly
  in release decision and release runbook status lines
- PR comments and signal lights now also expose release decision / runbook
  scorecard operator adoption and operational operator adoption, including
  outcome-drift summaries for both layers
- operator-adoption exporters now also compute `release_surface_alignment`,
  including mismatch lists between release decision and release runbook
- benchmark artifact bundle and companion summary now surface
  `release_surface_alignment` so downstream benchmark views do not need to open
  the standalone operator-adoption artifact
- `evaluation-report.yml`, job summary, and PR comments now surface operator
  release-surface alignment for the standalone operator-adoption surface and
  the bundle/companion downstream surfaces
- benchmark release surfaces now expose domain-level knowledge readiness for
  `tolerance`, `standards`, and `gdt`, including focus components and missing
  metrics
- `evaluation-report.yml`, job summary, and PR comments now also surface
  domain-level knowledge readiness counts, priority domains, and domain focus
  areas so release reviewers see standards/tolerance/GD&T gaps without opening
  the standalone benchmark artifacts
- benchmark knowledge drift now also exposes domain-level regressions,
  improvements, resolved domains, and new priority domains
- the same domain-drift signals now flow through benchmark artifact bundle,
  companion summary, release decision, and release runbook status surfaces

### 5. Vector / Qdrant Platformization

- coarse metadata contracts propagated into vector registration and search
- Qdrant-native search, list, similarity, compare, maintenance, stats, and
  migration surfaces delivered
- migration planning, readiness, pending listing, execution, and summary
  observability delivered

## Current Product Position

Relative to document-style CAD intelligence products, the platform now has a
stronger internal base in:

- engineering-semantic classification rather than OCR-only extraction
- review/reject/explain loops rather than single-shot prediction
- benchmark and governance artifacts rather than ad hoc evaluation
- vector backend operational visibility rather than opaque embedding storage

## Remaining Gaps

The main remaining gaps are no longer benchmark contract stability; they are
product depth:

- more real-data history-sequence validation with larger `.h5` sets
- more real-data STEP/B-Rep validation beyond smoke and small example batches
- richer standards/tolerance/GD&T checks surfaced directly in benchmark views
- richer operator/runbook automation that turns benchmark outputs into concrete
  execution batches
- broader real-data benchmarking that correlates domain readiness/drift with
  actual DXF, history-sequence, and STEP/B-Rep benchmark outcomes

## Recommended Next Build Order

1. Expand real-data validation reports for:
   - DXF hybrid benchmark runs
   - history-sequence `.h5` sets
   - STEP/B-Rep directory batches
2. Package a fuller operator-facing adoption loop that maps benchmark outputs
   to release, retraining, and review-queue execution.
3. Add benchmark-level real-data scorecards that compare weak-signal Graph2D,
   hybrid, OCR, history, and B-Rep surfaces in one report.
4. Extend domain-level knowledge readiness into CI-visible drift and release
   deltas after the standalone surfaces settle.
5. Add domain-aware benchmark views that tie standards/tolerance/GD&T status to
   real-data benchmark outcomes instead of only synthetic/operator summaries.

## Reference Docs

- `docs/COMPETITIVE_SURPASS_DESIGN_AND_DEVELOPMENT_20260308.md`
- `docs/BENCHMARK_ALIGNMENT_AND_BEYOND_STATUS_20260308.md`
- `docs/BENCHMARK_SURPASS_BUNDLE_PROGRESS_20260308.md`
- `docs/BENCHMARK_COMPANION_DELIVERY_PROGRESS_20260308.md`
- `docs/BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_VALIDATION_20260308.md`
- `docs/BENCHMARK_RELEASE_RUNBOOK_OPERATOR_ADOPTION_VALIDATION_20260308.md`
- `docs/BENCHMARK_RELEASE_SURFACE_OPERATOR_ADOPTION_CI_VALIDATION_20260308.md`
- `docs/BENCHMARK_RELEASE_SURFACE_OPERATOR_ADOPTION_PR_COMMENT_VALIDATION_20260308.md`
- `docs/BENCHMARK_RELEASE_SCORECARD_OPERATOR_ADOPTION_VALIDATION_20260309.md`
- `docs/BENCHMARK_RELEASE_SCORECARD_OPERATOR_ADOPTION_CI_VALIDATION_20260309.md`
- `docs/BENCHMARK_RELEASE_SCORECARD_OPERATOR_ADOPTION_PR_COMMENT_VALIDATION_20260309.md`
- `docs/BENCHMARK_OPERATOR_ADOPTION_RELEASE_ALIGNMENT_VALIDATION_20260309.md`
- `docs/BENCHMARK_OPERATOR_ADOPTION_RELEASE_ALIGNMENT_CI_VALIDATION_20260310.md`
- `docs/BENCHMARK_OPERATOR_ADOPTION_RELEASE_ALIGNMENT_PR_COMMENT_VALIDATION_20260310.md`
- `docs/BENCHMARK_OPERATOR_ADOPTION_RELEASE_SURFACES_VALIDATION_20260310.md`
- `docs/BENCHMARK_OPERATOR_ADOPTION_RELEASE_SURFACES_CI_VALIDATION_20260310.md`
- `docs/BENCHMARK_KNOWLEDGE_READINESS_VALIDATION_20260308.md`
- `docs/BENCHMARK_KNOWLEDGE_DOMAIN_READINESS_VALIDATION_20260308.md`
- `docs/BENCHMARK_KNOWLEDGE_READINESS_CI_VALIDATION_20260308.md`
