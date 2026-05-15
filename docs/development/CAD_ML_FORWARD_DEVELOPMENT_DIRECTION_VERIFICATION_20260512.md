# CAD ML Platform Forward Direction Verification

Date: 2026-05-12

## Repository State

Commands run:

```bash
git status --short
wc -l src/api/v1/analyze.py src/api/v1/vectors.py src/api/v1/dedup.py \
  src/api/v1/health.py src/api/v1/materials.py src/models/loader.py \
  src/api/v2/endpoints.py
rg -n "@router\\.(get|post|put|delete|patch)" src/api/v1 src/api/v2 \
  | cut -d: -f1 | sort | uniq -c | sort -nr | head -30
```

Results:

- Working tree was clean before adding these docs.
- `src/api/v1/analyze.py`: 164 lines.
- `src/api/v1/vectors.py`: 599 lines.
- `src/api/v1/dedup.py`: 1518 lines.
- `src/api/v1/health.py`: 1168 lines.
- `src/api/v1/materials.py`: 1146 lines.
- `src/models/loader.py`: 11 lines.
- `src/api/v2/endpoints.py`: 243 lines.
- Route decorator concentration remains highest in:
  - `health.py`: 31 route decorators.
  - `materials.py`: 23 route decorators.
  - `dedup.py`: 13 route decorators.

## Evidence Read

### Thin Analyze Router

`src/api/v1/analyze.py` delegates the main analyze path to
`run_analysis_live_pipeline(...)` and injects document, feature, vector, OCR,
classification, quality, and process pipeline functions. This supports the conclusion
that analyze router closeout is mostly successful.

### Phase 3 Vector Closeout

Existing documents read:

- `docs/development/PHASE3_VECTORS_LIST_ROUTER_DESIGN_20260429.md`
- `docs/development/PHASE3_VECTORS_BATCH_SIMILARITY_ROUTER_DESIGN_20260429.md`
- `docs/development/PHASE3_REMAINING_TODO_20260425.md`

Observed current state:

- `vectors_list_router.py` exists.
- `vectors_similarity_router.py` exists.
- `vector_list_models.py` exists.
- `vector_similarity_models.py` exists.
- `vectors.py` still includes routers and retains compatibility/helper ownership.
- `POST /backend/reload` still lives in `vectors.py`, so the admin router extraction is
  the next narrow Phase 3 closeout slice.

### Benchmark and Forward-Looking Gaps

`docs/BENCHMARK_ALIGNMENT_AND_BEYOND_STATUS_20260308.md` states:

- The platform has a benchmark-ready engineering baseline.
- Engineering, observability, governance, and migration partly exceed a simple
  benchmark-only system.
- `Graph2D`, `History Sequence`, and `3D/B-Rep` still lack enough real-data evidence.

This supports the roadmap stance that the architecture is forward-looking, while model
leadership claims need stronger scorecard evidence.

### 3D/B-Rep Evidence

Code read:

- `src/core/geometry/engine.py`
- `src/ml/vision_3d.py`
- `src/ml/pointnet/inference.py`
- `src/adapters/factory.py`

Findings:

- B-Rep graph extraction has explicit node and edge schema.
- UV-Net encoder scaffolding exists but falls back to mock embeddings when no checkpoint
  is available.
- PointNet analyzer returns `model_unavailable` fallback when no checkpoint is present.
- STEP/IGES adapter can use synthetic box geometry as a fallback, which is acceptable for
  demos but unsafe for benchmark evidence unless explicitly flagged.

### Model Readiness Evidence

`src/models/loader.py` is a static stub:

```python
_loaded = True
```

This supports the recommendation to implement a real model registry/readiness service
before making stronger model capability claims.

### API v2 Evidence

`src/api/v2/endpoints.py` contains response envelope and prediction concepts, but the
current API registration inspected in `src/api/__init__.py` is centered on `/v1`. The v2
prediction endpoints are still mock-like and should be treated as early scaffolding.

## Verification Outcome

The new roadmap intentionally separates:

- near-term contract-preserving architecture closeout,
- medium-term scorecard/model-readiness work,
- 3D/B-Rep real evidence work,
- later parametric/generative CAD work.

This matches the current code reality and avoids overclaiming the state of Graph2D,
UV-Net, PointNet, STEP/IGES parsing, and API v2.

