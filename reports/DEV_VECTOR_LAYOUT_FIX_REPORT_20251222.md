# DEV Vector Layout Fix Report (2025-12-22)

## Scope
- Normalize vector layout metadata (2D vs L3 tail)
- Preserve L3 tail during migration/preview
- Ensure preview uses target feature version
- Document layout tags

## Changes
- Added `src/core/vector_layouts.py` with layout constants and `layout_has_l3()` helper.
- `src/core/similarity.py` now defaults `vector_layout` to `base_sem_ext_v1` via shared constant.
- `src/api/v1/analyze.py`:
  - Registers vectors with explicit `feature_version`, `vector_layout`, `geometric_dim`, `semantic_dim`, `total_dim`.
  - Adds `l3_3d_dim` when 3D embedding appended and sets layout to `base_sem_ext_v1+l3`.
  - Ensures meta update uses the same payload and normalizes layout in legacy migrate path.
- `src/api/v1/vectors.py`:
  - Preview now uses `FeatureExtractor(feature_version=to_version)`.
  - Migration/preview split vectors using `_prepare_vector_for_upgrade()` to handle L3 tails and legacy layout.
  - Migration preserves L3 tail, updates `vector_layout`, dims, and `l3_3d_dim`.
- `docs/L4_FEATURES_USAGE.md` updated with layout tag definitions and L3 tail behavior.

## Behavior Notes
- L3 embedding vectors are tagged as `base_sem_ext_v1+l3`; migration upgrades/downgrades only the 2D portion and preserves the tail.
- Legacy layout (`geom_all_sem_v1`) is reordered to canonical format during migration before upgrade.

## Tests
- `.venv/bin/python -m pytest tests/unit/test_vector_migrate_layouts.py tests/unit/test_migration_preview_trends.py tests/unit/test_migration_preview_stats.py tests/unit/test_vector_migrate_v4.py -q`
- Result: `32 passed in 2.54s`
