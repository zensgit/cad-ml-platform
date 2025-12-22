#!/usr/bin/env markdown
# Feature Vector Layout Alignment Report

## Scope
- Align combined vector order with the v1-v4 slot definition (base geometric + semantic + extensions).
- Ensure cache storage, similarity registration, and migration use the same canonical layout.
- Add safety to vector upgrades when a current version is known.

## Changes
- Added `FeatureExtractor.flatten()` to build canonical vectors.
- Updated analyze flow to use flatten for feature cache, vector store, and ML classification input.
- Appended a cache key suffix (`layout_v2`) to avoid legacy cache reuse.
- Added `vector_layout` metadata on registered vectors.
- `upgrade_vector()` now accepts `current_version`; migration endpoints pass stored meta.

## Compatibility Notes
- Existing vectors stored with legacy layout (geometric + semantic) must be migrated or rebuilt
  before mixing with new vectors. This is required for memory/redis backends and any FAISS index.
- If a classification model was trained on the legacy layout, retrain it or regenerate vectors before use.

## Tests
- `.venv/bin/python -m pytest tests/unit/test_feature_rehydrate.py tests/unit/test_feature_vector_layout.py tests/unit/test_upgrade_vector.py tests/unit/test_feature_extractor_v4.py tests/unit/test_v4_feature_performance.py tests/unit/test_similarity_endpoint.py tests/unit/test_similarity_topk.py tests/unit/test_feature_slots.py -q`
- Result: `45 passed in 10.16s`
