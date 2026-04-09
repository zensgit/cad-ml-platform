# Benchmark Surpass Architecture

## Scope

- Baseline: `origin/main@19b3c6c4e9bebd639aef0dc38e85d26f3eea40f9`
- Date: `2026-03-08`
- Purpose: freeze the architecture used to benchmark against `Werk24`, `DrawInsight`, and similar engineering-document platforms.

## Architectural Position

This repository should be treated as an `engineering semantic platform`, not a single-model OCR service.

Primary product surfaces:

1. Classification and semantic normalization
2. Knowledge-backed explanation
3. Review and active-learning closure
4. Similarity/vector operations
5. Migration and operator governance

## Decision Hierarchy

The stable decision hierarchy is:

1. `hybrid`
2. `filename` / `titleblock`
3. `knowledge` confirmation / rejection / rerank
4. `graph2d` as weak signal
5. `history_sequence` and `3d/b-rep` as additive semantic branches

This means:

- `Graph2D` is no longer treated as the product primary classifier.
- `Hybrid` remains the dominant classification path.
- `Knowledge` is part of explanation and validation, not an opaque override.

## Stable Semantic Contract

Every major API/report/export surface should converge on:

- `fine_*`
- `coarse_*`
- `decision_source`
- `is_coarse_label`
- `source_contributions`
- `rejection_reason`

Current mainline progress already covers this contract across:

- analyze
- classify / batch-classify
- provider bridge
- vectors / similarity / compare
- feedback / active-learning
- evaluation and review-pack

## Explainability Layer

The explainability stack now has three operator-visible levels:

1. Model/branch explanation
   - `decision_path`
   - `source_contributions`
   - `hybrid_explanation`
2. Knowledge explanation
   - `knowledge_checks`
   - `violations`
   - `standards_candidates`
   - `knowledge_hints`
3. Assistant explanation
   - structured `evidence`
   - explainability helpers under `src/core/assistant/`

This is a major differentiation point versus benchmark tools that only expose extracted fields.

## Review and Feedback Closure

The production-safe loop is:

1. Detect weak / conflicting / low-confidence cases
2. Export review-pack with context
3. Collect feedback with coarse/fine normalization
4. Export retrain/metric inputs
5. Re-evaluate with history/benchmark scripts

Important operational outputs already in main:

- OCR review guidance
- OCR review pack export
- active-learning review queue
- review queue export
- feedback statistics
- finetune/metric observability

## Benchmark Scorecard Layer

The scorecard layer answers a different question from unit metrics:

- Is `hybrid` strong enough to act as primary?
- Is `graph2d` still weak-signal-only?
- Is `history_sequence` only smoke-tested or evidence-ready?
- Is `brep` prep-only or graph-ready?
- Is migration governance operational?

This architecture allows a single scorecard to summarize progress across:

- DXF real-data validation
- Graph2D diagnose/blind diagnose
- history evaluation
- STEP/B-Rep smoke or batch evaluation
- migration observability

## OCR Review Layer

OCR review is now modeled as a first-class operational layer:

- exportable review pack
- review priority
- primary gap
- automation readiness
- recommended actions

This is the point where the platform starts exceeding extraction-only benchmarks in operator usefulness.

## Vector / Qdrant Governance Layer

Benchmark tools rarely expose migration-safe vector operations. Mainline now includes:

- Qdrant-native list/search/similarity/topk/compare
- Qdrant-native register/update/delete
- stats/distribution observability
- migration status / trends / preview / pending / run / summary / readiness / plan

This is not cosmetic. It is what makes retrieval-backed AI maintainable in production.

## Long-Term Differentiators

The long-term surpass strategy remains:

1. `Hybrid + Knowledge + Review`
2. `Assistant evidence + operator explanations`
3. `History sequence`
4. `3D/B-Rep graph`

The repository does not need to beat authoring CAD systems at authoring. It needs to beat benchmark products at:

- semantic coherence
- rejection safety
- enterprise observability
- migration operability
- explainability

## Current Remaining Gaps

The architecture is in place, but the following still need more evidence:

- larger real-data `.h5` history validation
- broader STEP/B-Rep batch validation
- stronger benchmark score aggregation in CI using real artifacts
- more operator-facing summaries that combine scorecard + review + evidence

## Summary

The repository now has a coherent surpass architecture:

- `Hybrid` as primary
- `Graph2D` as weak signal
- `Knowledge` as structured explanation and validation
- `Review/feedback` as core product loop
- `Qdrant governance` as operator advantage
- `History/3D` as long-term semantic differentiators
