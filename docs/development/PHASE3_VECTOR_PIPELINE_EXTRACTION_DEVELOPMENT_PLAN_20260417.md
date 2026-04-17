# Phase 3 Vector Pipeline Extraction Development Plan

## Goal
- Extract the inline `vector registration + similarity dispatch` block from `src/api/v1/analyze.py` into a shared helper without changing the current metadata contract, backend routing, or response shape.

## Scope
- Move the analyze-path vector orchestration into `src/core/vector_pipeline.py`.
- Keep `analyze.py` responsible only for:
  - passing the current analysis context into the helper
  - writing returned similarity data back into `results`
  - recording stage timing and final request logging
- Preserve existing behavior for:
  - feature-vector flattening plus optional L3 embedding append
  - qdrant vs memory registration selection
  - optional FAISS mirroring when `VECTOR_STORE_BACKEND=faiss`
  - similarity computation when `calculate_similarity=true`
  - reference existence probe when only `reference_id` is provided

## Planned Changes
- Add `run_vector_pipeline(...)` to orchestrate:
  - `FeatureExtractor().flatten(features)`
  - optional L3 embedding append from `features_3d.embedding_vector`
  - `build_vector_registration_metadata(...)`
  - qdrant or memory registration
  - optional memory metadata enrichment
  - optional FAISS add-through
  - optional similarity dispatch to qdrant or memory backends
- Keep `_get_qdrant_store_or_none()` and `_compute_similarity_qdrant()` in `src/api/v1/analyze.py` and pass them into the helper as callbacks so route-level qdrant semantics stay unchanged.
- Replace the inline block in `src/api/v1/analyze.py` with one helper call.
- Add unit coverage for:
  - memory registration + metadata update path
  - qdrant registration + qdrant similarity path
  - registration failure that still allows similarity calculation
  - reference-not-found probe without similarity compute
  - FAISS mirror path
- Add an integration lock proving `/api/v1/analyze` now delegates vector work to the shared helper.

## Risk Controls
- Preserve the existing vector metadata contract by keeping `build_vector_registration_metadata(...)` unchanged.
- Preserve qdrant callback behavior by not moving `_get_qdrant_store_or_none()` or `_compute_similarity_qdrant()` in this slice.
- Preserve analyze response semantics:
  - only set `results["similarity"]` when the helper returns a payload
  - do not surface registration exceptions to the endpoint
- Keep stage timing in the caller so Prometheus labels and response timing remain unchanged.
