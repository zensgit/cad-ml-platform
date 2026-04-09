# Competitive Benchmark Surpass Execution Plan

Date: 2026-03-08
Status: Proposed
Scope: design and development plan only

Note: "Werk24", "DrawInsight", and "SOLIDWORKS-like" are used here as
workflow shorthand, not as exact vendor feature claims. The target is to beat
those workflow classes in the areas most relevant to this repository.

## Objective

Turn `cad-ml-platform` from a strong component platform into a benchmarked
engineering drawing intelligence product that can:

- beat Werk24-style structured drawing extraction
- beat DrawInsight-style drawing search, review, and insight workflows
- approach or exceed SOLIDWORKS-like design guidance for standards, fits,
  tolerances, and manufacturability assistance

The practical goal is not "more endpoints." It is a higher-confidence,
evidence-backed workflow from drawing input to structured output, retrieval,
review, and design guidance.

## Benchmark Targets

### 1. Werk24-style extraction workflow

Target outcome:

- ingest drawing image, PDF, DXF, DWG, and related package context
- extract title block, dimensions, tolerances, GD&T, notes, materials,
  standards references, tables, balloons, and sheet relationships
- return field-level confidence and provenance that is stable enough for ERP,
  MES, PLM, and review pipelines

Current repo relevance:

- `/api/v1/drawing/recognize`
- `/api/v1/ocr/*`
- drawing title-block parsing
- OCR golden evaluation and confidence calibration

### 2. DrawInsight-style retrieval and review workflow

Target outcome:

- semantic retrieval across drawings and derived engineering metadata
- duplicate and near-duplicate discovery
- filtered search by feature version, material, complexity, or standards hints
- review packs that surface conflicts, violations, standards candidates, and
  knowledge hints

Current repo relevance:

- `/api/v1/vectors/*`
- Qdrant and in-memory vector backends
- compare/review/reporting flows
- knowledge reporting artifacts

### 3. SOLIDWORKS-like guidance workflow

Target outcome:

- engineer asks for fit, tolerance, preferred dimension, surface finish, or
  standards guidance and gets a grounded answer
- system highlights design-rule violations and recommended next actions
- guidance is tied to extracted drawing evidence, not generic chat output

Current repo relevance:

- `/api/v1/tolerance/*`
- `/api/v1/standards/*`
- `/api/v1/design-standards/*`
- `/api/v1/assistant/*`
- provider-bridged knowledge modules

## Current Strengths

- Broad FastAPI surface already exists for `analyze`, `vision`, `ocr`,
  `drawing`, `vectors`, `tolerance`, `standards`, `design-standards`,
  `assistant`, `feedback`, `active-learning`, and `compare`.
- `drawing/recognize` already returns normalized title-block fields,
  dimensions, symbols, and field confidence.
- OCR evaluation already tracks dimension recall, symbol recall, edge F1,
  Brier score, and dual-tolerance accuracy.
- Vector CRUD, search, update, delete, batch similarity, migration preview,
  migration status, migration summary, migration pending, pending run, and
  migration trends already exist.
- Tolerance and fits knowledge is available through dedicated APIs.
- Standards and design-standards modules already exist and are exposed via the
  provider registry.
- Prometheus, runbooks, alert rules, eval history, baseline evaluation, and
  CI-oriented validation docs are already normal practice in this repo.

## Main Gaps

### Extraction gaps

- No single canonical drawing intelligence schema joins title block, views,
  dimensions, tolerances, GD&T, notes, tables, balloons, standards references,
  and provenance.
- Multi-sheet package reasoning, revision-aware extraction, and BOM/table
  linkage are not yet first-class.

### Retrieval and review gaps

- Retrieval infrastructure exists, but product-level semantic review workflows
  still look more like platform primitives than benchmark-ready analyst flows.
- Knowledge outputs are surfaced in reporting, but not yet consistently fused
  into every retrieval, compare, and review interaction.

### Guidance gaps

- Knowledge APIs exist, but drawing-aware guidance is still fragmented across
  assistant, standards, tolerance, and design-standard surfaces.
- There is no plugin-grade or authoring-loop experience yet for in-context
  design validation.

### Cross-cutting gaps

- Evidence chain and provenance are not yet consistent across extraction,
  retrieval, guidance, and review.
- Benchmark targets are not codified as a permanent CI gate.

## Phased Execution

### Phase 0: Benchmark contract and dataset

Deliver:

- one canonical drawing intelligence schema
- benchmark task set for extraction, retrieval, review, and guidance
- benchmark corpus split by difficulty and drawing type
- CI entry points that run benchmark subsets on relevant PRs

Exit criteria:

- benchmark schema frozen at `v1`
- labeled benchmark set available
- benchmark scripts produce machine-readable summary artifacts

### Phase 1: Extraction parity and confidence

Deliver:

- extend drawing extraction to GD&T, datums, weld/thread/surface-finish
  symbols, standards callouts, notes, revision block, tables, and BOM links
- unify OCR plus vision plus parser outputs under one evidence-backed response
- add multi-sheet PDF handling and package-level reconciliation

Exit criteria:

- extraction benchmark becomes the default gating signal for drawing changes
- outputs are stable enough to feed downstream review and retrieval

### Phase 2: Retrieval, compare, and review intelligence

Deliver:

- tie vectors, feature versions, and knowledge hints into one search/review
  payload
- make compare and review packs benchmark tasks instead of one-off tooling
- add "why this match" evidence: geometric features, OCR fields, standards
  hints, and knowledge checks

Exit criteria:

- top-k retrieval quality is benchmarked
- review artifact generation includes standards and tolerance context by
  default

### Phase 3: Design guidance and engineer copilot

Deliver:

- drawing-aware assistant answers grounded in standards, tolerance, and
  design-standard modules
- manufacturability and standards checks attached to extracted drawing content
- actionable recommendations for fit, tolerance class, preferred size, surface
  finish, and standards compliance

Exit criteria:

- assistant answers are grounded in extracted evidence and knowledge modules
- design guidance becomes part of review, not a side API

### Phase 4: Closed-loop learning

Deliver:

- connect feedback, active learning, retraining exports, and benchmark replay
- stratify learning loops by extraction mistakes, retrieval mistakes, and
  guidance mistakes
- use benchmark deltas and review outcomes to prioritize next improvements

Exit criteria:

- regression triage is driven by benchmark and review evidence
- accepted human corrections can flow into retraining-ready exports

## Acceptance Criteria

- stable canonical schema across extraction, retrieval, and review surfaces
- benchmark suites runnable in CI and locally
- evidence-backed responses for every major output type
- knowledge-guided review and guidance surfaced to users, not only internal
  logs
- closed-loop feedback path from review to export to retraining
