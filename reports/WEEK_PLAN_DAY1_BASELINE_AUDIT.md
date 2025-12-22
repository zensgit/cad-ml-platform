# Vector Store Layout Audit

- Timestamp: 2025-12-22T15:14:15.272337Z
- VECTOR_STORE_BACKEND: `memory`
- FEATURE_VERSION: `v1`

## FAISS Index
- Path: `data/faiss_index.bin`
- Exists: `False`

## Redis Vectors
- URL: `None`
- Reachable: `False`
- Error: `REDIS_URL not provided`
- Vector keys: `0`

## Migration Guidance
- No persisted Redis vectors detected (or Redis unreachable in this environment).
- No FAISS index file detected at the default path.
- If classification models consume vectors, retrain or regenerate features after layout change.

## Suggested Steps (Prod)
1. Freeze writes or route new writes to a fresh index.
2. Export IDs from Redis `vector:*` keys or analysis cache index.
3. Recompute vectors with `FeatureExtractor.flatten()` order and update meta `vector_layout`.
4. Rebuild FAISS index from migrated vectors and verify dimension consistency.
5. Run similarity regression tests and compare score distributions.
