# DEV Vector Pipeline Audit (2025-12-22)

## Scope
- Feature extraction + flatten/rehydrate layout
- Vector registration + meta layout tags
- Vector migration/upgrade paths
- Vector list/stats endpoints (memory/redis)

## Current Pipeline (2D)
- Extract: `FeatureExtractor.extract()` returns `{geometric, semantic}`
  - Base geometric (v1): entity_count, bbox_width, bbox_height, bbox_depth, bbox_volume_estimate
  - Semantic (v1): layer_count, complexity_high_flag
  - v2 extensions: norm_width, norm_height, norm_depth, width_height_ratio, width_depth_ratio
  - v3 extensions: solids_count, facets_count, avg_volume_per_entity, solids_ratio, facets_ratio, top_kind_freq_1..5
  - v4 extensions: surface_count, shape_entropy
- Flatten: `FeatureExtractor.flatten()` canonical order
  - `base_geometric(5) + semantic(2) + extensions(v2/v3/v4)`
- Cache: `feature_cache` stores flattened vectors keyed by `sha256(content)+FEATURE_VERSION+layout_v2`
  - Cache hit path uses `rehydrate()` to split into `geometric/semantic`
- Register: `analyze` uses `flatten(features)` and registers vector
  - `register_vector()` defaults meta: `feature_version`, `vector_layout=base_sem_ext_v1`
  - `analyze` enriches meta with `geometric_dim`, `semantic_dim`, `total_dim`, optional `l3_3d_dim`

## L3 3D Extension
- If 3D extraction succeeds, `embedding_vector` is appended to the 2D vector before register
- Meta still keeps `vector_layout=base_sem_ext_v1` (no explicit 3D layout tag)

## Migration / Upgrade Behavior
- `FeatureExtractor.upgrade_vector()` assumes canonical layout (base+semantic+extensions)
- Version inference is length-based when `current_version` is not explicitly provided
- `reorder_legacy_vector()` exists for legacy layout (geom_all + semantic) but is not called in migration paths

## Endpoints
- `/api/v1/vectors` lists from memory/redis; redis path scans `vector:*` and parses vector length from stored string
- `/api/v1/vectors_stats` summarizes counts and versions; redis path uses meta `feature_version` if present

## Risks / Gaps
- L3 embedding appended vectors exceed expected v1-v4 dims; migration/upgrade will fail on length checks
- `vector_layout` meta is always `base_sem_ext_v1`, even when L3 embedding is appended
- Legacy layout support exists (`reorder_legacy_vector`) but is not wired into migration or rehydrate

## Recommendations (next step)
- Tag layout explicitly when L3 embedding appended (e.g. `base_sem_ext_v1+l3`) and skip/handle during migration
- In migration, detect legacy layout via meta or length and use `reorder_legacy_vector` before `upgrade_vector`
- Document canonical order and L3 extension policy in `docs/L4_FEATURES_USAGE.md`

## Tests
- `.venv/bin/python -m pytest tests/unit/test_feature_vector_layout.py tests/unit/test_feature_rehydrate.py tests/unit/test_upgrade_vector.py tests/unit/test_upgrade_vector_v4_paths.py tests/unit/test_feature_extractor_v4.py tests/unit/test_feature_version_v3.py -q`
- Result: `18 passed in 5.16s`
