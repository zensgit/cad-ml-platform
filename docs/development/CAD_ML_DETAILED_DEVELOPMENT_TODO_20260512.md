# CAD ML Platform Detailed Development TODO

Date: 2026-05-12

## Summary

Goal: turn the current forward-looking CAD intelligence platform into a provable,
release-ready system. The execution order is fixed:

1. Finish Phase 3 architecture closeout.
2. Add real model readiness and fallback discipline.
3. Add a unified forward scorecard.
4. Build strict 3D/B-Rep evidence.
5. Introduce an explicit decision service.
6. Ground manufacturing intelligence in shared evidence.
7. Only then start parametric and generative CAD work.

Current branch:

```text
phase3-vectors-batch-similarity-router-20260429
```

## Phase 0: Baseline Documents

- [x] Add forward development direction doc.
- [x] Add forward direction verification doc.
- [x] Add this detailed TODO document.
- [ ] Keep these docs in the same PR as the first implementation slice unless the
  review gate asks for doc-only separation.

Acceptance:

- The docs identify architecture strengths, proof gaps, priority order, and validation
  boundaries.
- The docs do not claim Graph2D, UV-Net, PointNet, STEP/IGES, or API v2 are production
  leaders without evidence.

## Phase 1: Phase 3 Router Closeout

Goal: finish the remaining `vectors.py` router ownership cleanup without changing API
behavior.

- [x] Extract `POST /api/v1/vectors/backend/reload` into
  `src/api/v1/vectors_admin_router.py`.
- [x] Preserve final path:

```text
/api/v1/vectors/backend/reload
```

- [x] Preserve admin token dependency and auth-failure metric behavior.
- [x] Preserve response model name: `VectorBackendReloadResponse`.
- [x] Preserve `src.api.v1.vectors.*` compatibility exports:
  - `VectorBackendReloadResponse`
  - `_vector_reload_admin_token`
  - `reload_vector_backend`
  - `run_vector_backend_reload_pipeline`
- [x] Add route ownership guard for the admin route.
- [x] Add a broad guard that fails if future vector endpoints are added directly to
  `vectors.py`.
- [x] Pin backend reload route registration count, HTTP method, operationId,
  OpenAPI tag, response schema, and admin-token rejection in contract tests.
- [x] Pin vector list and batch-similarity split-router contracts, including route
  ownership, operationId, schema refs, response codes, and facade helper delegation.
- [x] Move pure vector filtering helper implementations into
  `src/core/vector_filtering.py` while preserving `src.api.v1.vectors.*` facade
  helper exports.
- [x] Move vector list source resolver into `src/core/vector_list_sources.py`
  while preserving the `src.api.v1.vectors._resolve_list_source` facade helper
  export.
- [x] Move vector migration config helper implementations into
  `src/core/vector_migration_config.py` while preserving
  `src.api.v1.vectors._resolve_vector_migration_scan_limit`,
  `src.api.v1.vectors._resolve_vector_migration_target_version`, and
  `src.api.v1.vectors._coerce_int` facade helper exports.
- [x] Move vector migration upgrade helper implementation into
  `src/core/vector_migration_upgrade.py` while preserving
  `src.api.v1.vectors._prepare_vector_for_upgrade` facade helper export.
- [x] Move vector migration readiness helper implementation into
  `src/core/vector_migration_readiness.py` while preserving
  `src.api.v1.vectors._build_vector_migration_readiness` facade helper behavior
  and facade target-version resolver patch compatibility.
- [x] Move Qdrant feature-version collection helper implementation into
  `src/core/vector_migration_qdrant_versions.py` while preserving
  `src.api.v1.vectors._collect_qdrant_feature_versions` facade helper behavior
  and facade scan-limit resolver patch compatibility.
- [x] Move Qdrant preview sample collection helper implementation into
  `src/core/vector_migration_qdrant_preview.py` while preserving
  `src.api.v1.vectors._collect_qdrant_preview_samples` facade helper export.
- [x] Move memory vector-list storage helper implementation into
  `src/core/vector_list_memory.py` while preserving
  `src.api.v1.vectors._list_vectors_memory` facade helper behavior and facade
  label-filter helper patch compatibility.
- [x] Move Redis vector-list storage helper implementation into
  `src/core/vector_list_redis.py` while preserving
  `src.api.v1.vectors._list_vectors_redis` facade helper behavior and facade
  label-filter helper patch compatibility.
- [x] Move Qdrant vector-list storage branch implementation into
  `src/core/vector_list_qdrant.py` while preserving
  `src.core.vector_list_pipeline.run_vector_list_pipeline` source resolution,
  filter-builder injection, and extractor patch compatibility.
- [x] Move vector-list limit resolution into
  `src/core/vector_list_limits.py` while preserving `VECTOR_LIST_LIMIT`
  clamping and `VECTOR_LIST_SCAN_LIMIT` pass-through semantics.
- [x] Move Qdrant pending-candidate collection branch into
  `src/core/vector_migration_pending_qdrant.py` while preserving
  `src.core.vector_migration_pending_candidates.collect_vector_migration_pending_candidates`
  as the shared memory/Qdrant entrypoint.
- [x] Move memory pending-candidate collection branch into
  `src/core/vector_migration_pending_memory.py` while preserving
  `src.core.vector_migration_pending_candidates.collect_vector_migration_pending_candidates`
  as the shared normalization and dispatch entrypoint.
- [ ] Continue remaining shared helper ownership cleanup only in small slices with
  facade compatibility tests.

Acceptance:

- OpenAPI operationId remains:

```text
reload_vector_backend_api_v1_vectors_backend_reload_post
```

- Existing monkeypatch tests against `src.api.v1.vectors.run_vector_backend_reload_pipeline`
  continue to pass.
- `vectors.py` becomes a compatibility facade plus router aggregator for backend reload.

## Phase 2: Model Readiness Registry

Goal: replace static readiness with true model state.

- [x] Replace `src/models/loader.py` static `_loaded = True` behavior.
- [x] Add a model readiness registry with entries for:
  - V16 classifier
  - Graph2D
  - UV-Net
  - PointNet
  - OCR provider
  - embedding model
- [x] Report per-model:
  - `enabled`
  - `checkpoint_exists`
  - `loaded`
  - `version`
  - `checksum`
  - `fallback_mode`
  - `error`
- [x] Update `/ready` and health endpoints to use registry state.
- [x] Keep local development runnable without checkpoints, but mark missing models as
  degraded or fallback.
- [ ] Connect each lazy model implementation to explicit loaded/error callbacks as the
  classifiers move from availability checks to eager production load checks.
- [ ] Add load-failure fixtures for Graph2D/UV-Net/PointNet checkpoint incompatibility.

Acceptance:

- No checkpoint means no "real model ready" status.
- Readiness distinguishes disabled, fallback, degraded, and loaded.
- Unit tests cover checkpoint present, missing, required-missing blocking, and fallback.
- Load-failure coverage remains a follow-up because this slice does not eagerly load
  heavyweight checkpoint classes.

## Phase 3: Unified Forward Scorecard

Goal: make "are we ahead?" answerable by one artifact.

- [x] Add a scorecard exporter.
- [x] Aggregate:
  - Hybrid DXF coarse accuracy and macro F1.
  - Graph2D blind accuracy and low-confidence rate.
  - History Sequence coarse/fine metrics.
  - B-Rep parse and graph extraction metrics.
  - Qdrant/vector migration readiness.
  - Active-learning review queue health.
  - Knowledge grounding coverage.
  - Manufacturing evidence coverage.
- [x] Emit:

```text
reports/benchmark/forward_scorecard/latest.json
reports/benchmark/forward_scorecard/latest.md
```

- [x] Use status values:
  - `release_ready`
  - `benchmark_ready_with_gap`
  - `shadow_only`
  - `blocked`
- [x] Wire CI/release jobs to pass real benchmark artifact inputs into
  `scripts/export_forward_scorecard.py`.
- [x] Add scheduled/PR checks that fail release labels when the scorecard is
  `blocked` or `shadow_only`.

Acceptance:

- Release claims must cite this scorecard.
- Fallback-only results cannot be `release_ready`.
- 3D/B-Rep is reported separately from stronger 2D Hybrid DXF results.

## Phase 4: Strict STEP/IGES and B-Rep Golden Set

Goal: turn B-Rep from scaffolding into measurable capability.

- [x] Add strict evaluation mode for STEP/IGES.
- [x] Prevent invalid STEP/IGES from becoming synthetic-box benchmark success.
- [x] Keep synthetic geometry only for explicitly marked demo mode.
- [x] Add a STEP/IGES golden manifest contract and validator.
- [x] Add a non-release example manifest fixture.
- [x] Make strict B-Rep directory evaluation reproducible from a manifest.
- [x] Wire the B-Rep golden manifest validator into the optional CI/release flow.
- [x] Wire manifest-driven strict B-Rep evaluation into the forward scorecard inputs.
- [ ] Populate the manifest with 50-100 real, release-eligible STEP/IGES files.
- [x] Track:
  - parse success
  - face count
  - edge count
  - solid count
  - surface type histogram
  - graph validity
  - extraction latency
  - failure reason
- [x] Add B-Rep graph extraction QA report.
- [x] Feed B-Rep metrics into the forward scorecard.

Acceptance:

- Synthetic geometry is never counted as real benchmark success.
- Every failed sample has a stable failure reason.
- Golden reports are reproducible from the manifest.
- The manifest validator must fail release readiness when fewer than 50 real,
  release-eligible samples are present.
- CI can keep the manifest gate disabled for ordinary PRs, but when enabled it must
  publish validation evidence and fail release readiness on an insufficient golden set.

## Phase 5: DecisionService Boundary

Goal: centralize final CAD decisions and make every signal auditable.

- [x] Add a `DecisionService` design doc.
- [x] Implement a minimal service that consumes:
  - baseline classifier
  - filename/titleblock/OCR signals
  - Graph2D shadow signal
  - B-Rep hints
  - knowledge checks
  - vector-neighbor evidence
  - active-learning history
- [x] Return a stable contract:
  - `fine_part_type`
  - `coarse_part_type`
  - `confidence`
  - `decision_source`
  - `branch_conflicts`
  - `evidence`
  - `review_reasons`
  - `fallback_flags`
  - `contract_version`
- [x] Route the analyze classification pipeline through `DecisionService`.
- [x] Route batch classify through `DecisionService`.
- [x] Route assistant explanation through the shared decision evidence contract.
- [x] Route benchmark exporters through the shared decision evidence contract.

Acceptance:

- Analyze, batch classify, assistant explanation, and benchmark exporters use the same
  final decision contract.
- New model branches do not require API router changes.
- Low-confidence and conflict cases enter the review queue with evidence.

## Phase 6: Knowledge-Grounded Manufacturing Intelligence

Goal: extend value beyond classification.

- [x] Connect material, tolerance, process, cost, and DFM checks into analyze evidence.
- [x] Add rule source/version metadata to every recommendation.
- [x] Make assistant answers cite the same structured evidence used by analyze.
- [x] Add fixtures for:
  - [x] material substitution
  - [x] H7/g6 fit validation
  - [x] surface finish recommendation
  - [x] machining process route
  - [x] manufacturability risk
- [x] Add knowledge grounding coverage to the forward scorecard.
- [x] Add manufacturing evidence coverage to the forward scorecard.
- [x] Emit forward-scorecard-compatible manufacturing evidence summaries from real
  DXF benchmark analyze outputs.
- [x] Publish one real manufacturing evidence benchmark artifact from CI/release
  workflow variables.
- [x] Add reviewed source correctness metrics for DFM/process/cost/decision evidence.
- [x] Extend correctness labels from source presence to top-level payload quality.
- [x] Extend payload quality labels into nested source-specific details.
- [x] Add a review manifest generator and minimum-label validator for source,
  payload, and detail labels.
- [x] Wire the review manifest validator into the optional CI/release scorecard flow.
- [x] Surface review manifest validation status and artifact path in the forward
  scorecard.
- [x] Enforce approved review status and optional reviewer metadata before manifest
  rows count as release-reviewed labels.
- [x] Add an approved-only merge path from review manifest rows back into the
  benchmark manifest.
- [x] Wire the approved-only reviewed benchmark manifest merge into the optional
  CI/release scorecard artifact flow.
- [x] Make hybrid blind evaluation prefer the merged reviewed benchmark manifest
  when the scorecard merge output is available.
- [x] Add an optional hybrid blind gate requirement that release runs must consume
  the merged reviewed benchmark manifest.
- [x] Emit an actionable manufacturing review progress Markdown report for reviewer
  closeout and CI artifacts.
- [x] Emit a machine-readable manufacturing review gap CSV for reviewer assignment
  and closeout tracking.
- [x] Emit a label-grouped manufacturing review assignment Markdown plan for
  reviewer batching.
- [x] Emit a reviewer fill-template CSV for the remaining manufacturing review
  patch rows.
- [x] Add a reviewer-template apply path to patch approved reviewer rows back into
  the full review manifest.
- [x] Wire optional reviewer-template apply into the forward scorecard CI flow before
  review manifest validation.
- [x] Add a reviewer-template preflight validator for local and CI self-checks.
- [x] Wire reviewer-template preflight into optional forward scorecard CI before
  template apply.
- [x] Emit reviewer-template preflight Markdown for human-readable CI artifacts.
- [x] Emit manufacturing review handoff Markdown to package reviewer closeout
  artifacts and commands.
- [x] Emit reviewer-template preflight gap CSV for machine-readable closeout
  tracking after filled-template return.
- [x] Emit reviewer-template apply audit CSV for machine-readable post-apply
  outcome tracking.
- [x] Emit reviewed-manifest merge audit CSV for machine-readable benchmark merge
  outcome tracking.
- [x] Emit manufacturing review context CSV for reviewer evidence lookup during label
  closeout.
- [x] Emit manufacturing review batch CSV for label-balanced reviewer worklists.
- [x] Emit manufacturing review batch template CSV for bounded reviewer label entry.
- [x] Decouple reviewer-template preflight minimum rows from release validation
  minimums for partial batch returns.
- [x] Surface partial reviewer-template preflight thresholds in review handoff
  commands.
- [x] Prefer batch reviewer templates in generated review handoff preflight/apply
  commands when a bounded batch template exists.
- [x] Make reviewer-template preflight verify returned rows against the current
  review manifest before apply.
- [x] Make reviewer-template preflight report an empty base manifest as an
  explicit artifact blocker.
- [x] Make reviewer-template preflight report duplicate base manifest row
  identities before apply.
- [x] Block direct reviewer-template apply when the base review manifest has
  duplicate row identities.
- [x] Block approved review-manifest merge when the base benchmark manifest has
  duplicate row identities.
- [x] Block approved review-manifest merge when file-name fallback would be
  ambiguous across duplicate base file names.
- [x] Block direct reviewer-template apply when file-name fallback would be
  ambiguous across duplicate base file names.
- [x] Make reviewer-template preflight block file-name fallback when the base
  review manifest has duplicate file names.
- [x] Cover ambiguous file-name preflight blocking end-to-end through CLI
  summary, Markdown, and gap CSV artifacts.
- [ ] Populate real reviewed source, payload, and detail labels for the release
  benchmark set.
- [ ] Tune source, payload, and detail quality thresholds after the release review set
  is stable.

Acceptance:

- Analyze and assistant outputs agree on rule source/version.
- Missing knowledge coverage degrades scorecard status.
- Engineering recommendations are explainable, not just text output.
- Manufacturing intelligence release claims cite benchmark evidence coverage, not
  only fixture-level behavior.

## Phase 7: Parametric and Generative CAD

Goal: move from recognizing CAD to recovering, editing, and generating CAD.

Prerequisite: Phases 2-4 are credible and scorecard-backed.

- [ ] Continue History Sequence dataset and evaluator.
- [ ] Define a parametric command schema or CadQuery output contract.
- [ ] Add reverse-CAD evaluation:
  - drawing to editable CAD
  - point cloud to editable CAD
  - mesh to editable CAD
- [ ] Track:
  - geometric error
  - topology validity
  - constraint validity
  - editability
  - manufacturability
- [ ] Label Text-to-CAD as experimental until scorecard-backed.

Acceptance:

- Outputs are editable parameterized representations, not only meshes.
- Generated CAD passes geometry and manufacturability checks.
- Experimental demos are not described as production capability.

## Immediate Next PRs

| Priority | PR | Scope |
| --- | --- | --- |
| P0 | `phase3-vectors-admin-router` | Finish backend reload route extraction and route ownership tests |
| P0 | `model-readiness-registry` | Replace static model readiness and expose fallback status |
| P1 | `forward-scorecard-v1` | Add unified benchmark/readiness scorecard artifacts |
| P1 | `brep-strict-eval` | Strict STEP/IGES evaluation and B-Rep golden manifest |
| P2 | `decision-service-v1` | Centralize final CAD decision contract |

## Validation Commands

Use targeted validation first:

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  src/api/v1/vectors.py \
  src/api/v1/vectors_admin_router.py \
  tests/unit/test_vectors_admin_router.py

pytest -q \
  tests/unit/test_vectors_admin_router.py \
  tests/unit/test_vectors_backend_reload_delegation.py \
  tests/unit/test_vectors_backend_reload_admin_token.py \
  tests/unit/test_vector_backend_reload_failure.py \
  tests/contract/test_openapi_operation_ids.py \
  tests/contract/test_openapi_schema_snapshot.py
```

Run broader validation before merge:

```bash
pytest -q tests/unit/test_vectors_*.py tests/unit/test_vector_backend_reload_*.py
pytest -q tests/contract/test_openapi_operation_ids.py tests/contract/test_openapi_schema_snapshot.py
```
