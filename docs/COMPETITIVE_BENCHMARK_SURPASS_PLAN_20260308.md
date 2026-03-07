# Competitive Benchmark Surpass Plan

## Benchmark Targets

Primary comparison targets for this repository are:

- `Werk24`
  - Strong in drawing extraction, title block parsing, dimension/GD&T structuring, and stable API output.
- `DrawInsight`
  - Strong in enterprise workflow automation, part-family classification, and drawing-to-business-object mapping.
- `SOLIDWORKS Drawings/MBD` class workflows
  - Strong in authoring-time assistance, annotation generation, and in-CAD productivity.

This project should not try to out-author CAD authoring systems. It should instead outperform them in:

- private deployment
- hybrid AI + rules + knowledge fusion
- rejection and review safety
- enterprise-specific taxonomy and standards alignment
- cross-source understanding across DXF, STEP/B-Rep, title block, filename, history, vectors, and review feedback

## Current Strengths in This Repository

The repository already contains foundations many benchmark products do not combine in one stack:

- `HybridClassifier` as the primary AI decision surface
- `Graph2D` as a weak geometric signal and diagnostics path
- filename/titleblock/history/process branches
- coarse/fine label normalization across multiple API surfaces
- review-pack / active-learning export chain
- vector search, similarity, compare, and Qdrant-backed metadata contracts
- tolerance / GD&T / standards / design-standards knowledge modules
- STEP/B-Rep extraction path and graph dataset work
- CI gates for report generation, review strictness, and observability

This means the strategic gap is no longer “missing AI modules”. The gap is product coherence:

- stable output contracts
- better operator observability
- stronger knowledge-backed decisions
- better enterprise feedback loops

## Product Positioning

The target product position should be:

`Engineering Semantic Platform`, not just `drawing OCR` or `single-model classifier`.

The platform should answer five classes of questions:

1. What is this drawing / part at coarse and fine levels?
2. Why was that decision made?
3. What knowledge or standards constraints apply?
4. Is the result safe enough to automate, or should it be rejected/reviewed?
5. How do operators migrate, observe, and maintain model/vector state safely?

## Surpass Strategy

### 1. Stable Output Wins First

Benchmarks win because their API outputs are stable. This repository now needs to standardize:

- `fine_*`
- `coarse_*`
- `decision_source`
- `is_coarse_label`
- `rejection_reason`
- `source_contributions`

Winning condition:

- the same sample returns consistent semantics across analyze, classify, vectors, compare, similarity, feedback, evaluation, and review-pack surfaces

### 2. Hybrid Beats Single-Model

The long-term winning architecture remains:

- `Hybrid` as the default final classifier
- `Graph2D` as weak signal / conflict signal / reject signal
- `knowledge` as confirm / reject / rerank / explain
- `history` and `3D/B-Rep` as auxiliary semantic signals

Winning condition:

- the product no longer depends on any single model branch to stay useful

### 3. Knowledge Becomes Product, Not Sidecar

Competitors often extract fields. This project can exceed them by validating engineering meaning:

- tolerance structure
- GD&T structure
- standards candidates
- process/material consistency
- drawing-rule violations

Winning condition:

- outputs explain not only classification but engineering plausibility

### 4. Review and Feedback Are Core Product Features

Industrial AI wins by safe rejection and correction loops, not by forcing an answer.

Winning condition:

- low-confidence and conflict cases flow into review with enough context
- operator feedback becomes retrain/evaluation input with minimal manual remapping

### 5. Operator-Grade Vector Operations Matter

To surpass internal tooling and benchmark enterprise products, vector operations must become operable:

- Qdrant-native query/list/search/update/delete
- migration status, preview, pending, plan, run, recommendations
- coverage/truncation/readiness summaries

Winning condition:

- a maintainer can understand migration safety without reading raw vector-store internals

## Implementation Phases

### Phase A: Stable Semantics

Scope:

- complete coarse/fine normalization everywhere
- ensure provider, similarity, vector search, compare, feedback, and batch classify surfaces expose the same semantic contract

Success metrics:

- no API surface returns only an ambiguous `category` without coarse/fine context where semantics are expected
- evaluation scripts report both exact and coarse metrics

### Phase B: Knowledge-Backed Decisions

Scope:

- expose `knowledge_checks`, `violations`, `standards_candidates`, and structured summaries
- add knowledge-aware review and evaluation summaries

Success metrics:

- review/export surfaces include knowledge conflict reasons
- final decision remains explainable after knowledge involvement

### Phase C: Operable Vector Platform

Scope:

- Qdrant-native contracts across list/search/similarity/compare/stats/maintenance/migration
- migration plan/readiness/coverage/truncation/runbook surfaces

Success metrics:

- operators can safely answer:
  - what is pending
  - what should run first
  - what is blocked
  - how much is covered by the current plan

### Phase D: Review and Feedback Loop

Scope:

- review-pack enrichment
- feedback statistics
- finetune/export observability
- drift and correction analysis

Success metrics:

- correction volume and correction type can be measured by coarse/fine label and decision source

### Phase E: 3D and History as Long-Term Differentiators

Scope:

- history real-data validation
- STEP/B-Rep smoke and batch validation
- feature-hint integration

Success metrics:

- 3D/history become additive semantic branches, not blocking dependencies

## Acceptance Criteria

The repository can claim a meaningful “surpass trajectory” when these conditions hold:

- `Hybrid` is the obvious primary decision path in real validation
- `Graph2D` is productized as weak signal and diagnostics, not misrepresented as the main classifier
- review/export/feedback/evaluation all preserve coarse/fine semantics and decision source
- vector operations are safe and operator-readable
- at least one real-data DXF validation and one real-data `.h5`/STEP validation path are documented and replayable
- knowledge outputs participate in explanation and review

## Merge and Delivery Guidance

Preferred merge grouping:

1. semantic contract stack
2. review/feedback/export stack
3. Qdrant native data-plane stack
4. migration operability stack
5. docs and rollout notes

Do not interleave unrelated wide-surface changes into these stacks. Keep merge batches coherent and replayable.

## What Not To Optimize First

Do not prioritize these ahead of stable product semantics:

- another standalone Graph2D model rewrite
- broad 3D feature-recognition ambitions without operator-visible value
- large UI work before API/review/export contracts stabilize

## Final Direction

The benchmark-surpass goal should be defined as:

`More explainable, more enterprise-adaptable, more safely automatable engineering semantics`

not:

`one model with a slightly better raw benchmark score`.
