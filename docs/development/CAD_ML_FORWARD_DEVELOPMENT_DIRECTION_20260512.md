# CAD ML Platform Forward Development Direction

Date: 2026-05-12

## Scope

This document turns the current code reading into an execution-oriented development
direction. It is based on the local checkout on branch
`phase3-vectors-batch-similarity-router-20260429`.

The core conclusion is:

- The platform direction is forward-looking at the architecture level.
- The strongest current asset is not one model, but the engineering loop around CAD
  analysis, decision contracts, vector migration, review, benchmark, and release gates.
- The main gap is proof: real-data scorecards, model readiness, and 3D/B-Rep evidence
  are not yet strong enough to claim comprehensive model leadership.

## Current Position

| Area | Current state | Direction quality | Main risk |
| --- | --- | --- | --- |
| API and pipeline architecture | `analyze.py` is a thin orchestration facade; major logic has moved to core pipelines | Strong | Several large routers still remain |
| Classification decision path | Hybrid/fusion/shadow/review pipeline exists and keeps Graph2D as weak signal | Strong | Decision service boundary is still implicit |
| Vector platform | Qdrant migration path, migration plans, metadata filters, and compatibility facades exist | Strong | `vectors.py` still owns helper compatibility and backend reload |
| Benchmark and release governance | Many benchmark exporters and release-surface gates exist | Strong | Results are fragmented across many artifacts |
| 3D/B-Rep | OpenCascade/B-Rep graph schema and UV-Net scaffolding exist | Promising | Real model/data evidence is weak; fallback paths can hide gaps |
| Point cloud | PointNet API exists with deterministic fallback | Useful but not differentiating | No checkpoint means no real model claim |
| Assistant and knowledge | Domain assistant, retrieval, materials/tolerance/process knowledge are present | Promising | Needs stronger grounding into analyze evidence and scorecards |
| API v2 | Envelope and versioning idea exists | Early | Current endpoints are still mock-like and not registered as primary API |

## Strategic Thesis

The winning direction should be:

1. Treat CAD intelligence as a decision system, not a single classifier.
2. Make `Hybrid` the production path and keep weak modalities as evidence, conflict
   signals, and active-learning input.
3. Move the next differentiating investment from 2D Graph2D accuracy claims into
   real B-Rep understanding and editable/parametric CAD recovery.
4. Make every model claim pass through a unified real-data scorecard before it becomes
   release language.
5. Prefer small contract-preserving PRs until the remaining router and compatibility
   work is closed.

## Phase A: Finish Architecture Closeout

Goal: finish the remaining Phase 3 router and compatibility closeout without changing
behavior.

Recommended PRs:

1. `vectors` backend reload admin router extraction.
   - Move `POST /api/v1/vectors/backend/reload` into `vectors_admin_router.py`.
   - Preserve admin token dependency, auth-failure metrics, response schema, and
     operationId.
   - Keep `src.api.v1.vectors.*` compatibility exports.

2. Vector helper ownership guard.
   - Keep compatibility facade in `vectors.py`.
   - Move shared filter/list/search helpers only after route ownership is stable.
   - Add a test that fails if new vector endpoints are added directly to `vectors.py`.

3. Large-router triage.
   - Do not immediately refactor `health.py`, `materials.py`, or `dedup.py`.
   - First create ownership maps: route group, response model location, pipeline owner,
     external compatibility risk.

Acceptance:

- `analyze.py` remains a thin orchestrator.
- `vectors.py` has no direct endpoint except compatibility-only exports.
- OpenAPI path, operationId, and response schemas remain stable.
- Existing monkeypatch surfaces remain compatible.

## Phase B: Unified Evidence Scorecard

Goal: make "are we ahead?" answerable by one artifact.

Build a single scorecard that aggregates:

- Hybrid DXF coarse accuracy, macro F1, final decision source distribution.
- Graph2D blind accuracy, low confidence rate, branch conflict contribution.
- History Sequence coarse/fine metrics and mismatch summary.
- B-Rep/3D valid parse rate, graph extraction rate, model status, embedding stability.
- Qdrant/vector migration readiness.
- Active learning review queue volume, evidence quality, closure rate.
- Knowledge grounding coverage for material, tolerance, standards, process, cost.

Recommended outputs:

- `reports/benchmark/forward_scorecard/latest.json`
- `reports/benchmark/forward_scorecard/latest.md`
- CI summary line with one of:
  - `release_ready`
  - `benchmark_ready_with_gap`
  - `shadow_only`
  - `blocked`

Acceptance:

- No release claim can cite model accuracy without the scorecard artifact.
- Graph2D can only be promoted if it passes blind and real-data gates.
- 3D/B-Rep is reported separately from 2D DXF so it cannot be hidden by stronger
  Hybrid DXF results.

## Phase C: Model Readiness and Fallback Discipline

Goal: stop fallback behavior from being mistaken for real model capability.

Work items:

- Replace the stub model loader with a model registry/readiness service.
- Report checkpoint presence, version, checksum, loaded status, and fallback status.
- Mark UV-Net, PointNet, Graph2D, OCR provider, and V16 classifier readiness separately.
- Add "fallback used" and "synthetic geometry used" fields to analysis and scorecard
  payloads.
- Make benchmark scripts fail or mark `blocked` when synthetic/fallback model paths are
  used unintentionally.

Acceptance:

- `/ready` does not report model readiness from a static `_loaded = True`.
- Evaluation artifacts distinguish `model_unavailable`, `mock_embedding`,
  `synthetic_geometry`, and `real_model`.
- Public or stakeholder-facing claims cannot be generated from fallback-only runs.

## Phase D: 3D/B-Rep Real Data Track

Goal: turn B-Rep from promising scaffolding into a measurable differentiator.

Recommended slices:

1. Strict STEP/IGES parse mode.
   - In evaluation mode, invalid STEP/IGES must not produce synthetic box geometry.
   - Keep synthetic fallback only for local demo paths, with explicit status.

2. B-Rep graph golden set.
   - Build a small golden manifest first: 50-100 STEP/IGES files.
   - Track parse success, face/edge/solid counts, surface type histogram, graph validity,
     extraction time, and failure reason.

3. UV-Net/BRepNet-style training gate.
   - Train only after graph extraction quality is stable.
   - Keep model comparison separate from PointNet/mesh baselines.

4. 3D similarity contract.
   - Store B-Rep embeddings in Qdrant with metadata:
     `format`, `feature_version`, `brep_schema_version`, `model_version`,
     `fallback=false`.

Acceptance:

- B-Rep benchmark has reproducible manifests and failure taxonomy.
- Real model inference is distinguished from heuristic hints.
- 3D path can answer: "understands topology" vs "only sees a mesh/point cloud".

## Phase E: Explicit Decision Service

Goal: centralize final CAD decisions and make every signal auditable.

Create a `DecisionService` boundary that consumes:

- baseline classifier result
- filename/titleblock/OCR signals
- Graph2D shadow signal
- B-Rep/3D hints or embedding model output
- material/tolerance/process/knowledge checks
- vector-neighbor evidence
- active-learning history

It should return:

- `fine_part_type`
- `coarse_part_type`
- `confidence`
- `decision_source`
- `branch_conflicts`
- `evidence`
- `review_reasons`
- `fallback_flags`
- `contract_version`

Acceptance:

- Analyze, batch classify, assistant explanations, and benchmark scripts use the same
  decision contract.
- New model branches can be added without changing API router code.
- Low-confidence and conflict cases automatically enter review queue with evidence.

## Phase F: Knowledge-Grounded Manufacturing Intelligence

Goal: make the platform useful beyond classification.

Development direction:

- Connect material, tolerance, process, cost, and DFM checks into analyze evidence.
- Make assistant answers cite the same structured evidence used by analyze.
- Add rule source/version metadata for every recommendation.
- Build fixtures for common engineering questions:
  - material substitution
  - H7/g6 fit validation
  - surface finish recommendation
  - machining process route
  - manufacturability risk

Acceptance:

- Analyze can explain why a part was classified and why a process/material was
  recommended.
- Assistant answers and API outputs agree on the same rule source and version.
- Knowledge coverage is part of the scorecard, not a separate nice-to-have.

## Phase G: Parametric and Generative CAD Direction

Goal: move from "recognize CAD" to "recover, edit, and generate CAD".

This should start only after Phases B-D are credible.

Recommended track:

1. CAD command sequence dataset and evaluator.
   - Continue History Sequence work.
   - Add command validity, reconstruction error, and editability metrics.

2. Reverse engineering from 2D/3D.
   - Point cloud or drawing to parametric sketch/code.
   - Output CadQuery/Python or an internal neutral command schema.

3. Text-to-CAD or engineer-assistant CAD generation.
   - Keep it gated as experimental.
   - Require manufacturability and standards validation before any "production" claim.

Acceptance:

- Generated/recovered CAD is editable, not just mesh-like output.
- Evaluation includes geometric error, topology validity, constraint validity, and
  downstream manufacturability checks.

## Priority Order

| Priority | Work | Reason |
| --- | --- | --- |
| P0 | Finish Phase 3 `vectors` admin/router ownership closeout | Low-risk architecture debt, blocks clean future changes |
| P0 | Model readiness registry and fallback flags | Prevents overclaiming and false readiness |
| P1 | Unified forward scorecard | Turns direction into measurable progress |
| P1 | Strict STEP/IGES evaluation mode | Protects 3D/B-Rep evidence quality |
| P1 | B-Rep golden manifest and graph extraction QA | Required before real 3D model investment |
| P2 | DecisionService boundary | Makes multi-signal intelligence maintainable |
| P2 | Knowledge-grounded manufacturing evidence | Differentiates product value from classifier demos |
| P3 | Parametric/reverse-CAD generation | High-upside, but only after evidence foundations |

## Work To Avoid For Now

- Do not market Graph2D as the primary model until blind/real-data metrics support it.
- Do not refresh OpenAPI snapshots to silence failures unless the API change is intended.
- Do not let synthetic geometry or mock embeddings enter benchmark results as success.
- Do not start broad large-router refactors before the vector closeout stack lands.
- Do not build text-to-CAD demos before the scorecard can label them experimental.

