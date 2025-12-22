# Vector Store Layout Audit

- Timestamp: 2025-12-22T13:27:49.212654Z
- VECTOR_STORE_BACKEND: `redis`
- FEATURE_VERSION: `v1`

## FAISS Index
- Path: `data/faiss_index.bin`
- Exists: `False`

## Redis Vectors
- URL: `redis://localhost:16379/0`
- Reachable: `True`
- Vector keys: `8`

### Sample Vectors
- `vector:19bd5691-877d-4cea-9341-7bcd9ccc7590` dim=7 version=v1 layout=base_sem_ext_v1 ts=1766410004
- `vector:6ec7f3cc-1a82-432e-b83e-9833de3d4ed1` dim=7 version=v1 layout=base_sem_ext_v1 ts=1766393978
- `vector:026d5ae0-3f6c-4242-bbc5-bcd5b5716b28` dim=7 version=v1 layout=base_sem_ext_v1 ts=1766394483
- `vector:b22e9516-8464-4546-bc44-da52eb4fe05c` dim=7 version=v1 layout=base_sem_ext_v1 ts=1766410057
- `vector:609517f4-4cc8-4e88-be86-f239fd0f5c34` dim=7 version=v1 layout=base_sem_ext_v1 ts=1766410004

## Migration Guidance
- Redis contains persisted vectors; plan a full migration to the new layout.
- No FAISS index file detected at the default path.
- If classification models consume vectors, retrain or regenerate features after layout change.

## Suggested Steps (Prod)
1. Freeze writes or route new writes to a fresh index.
2. Export IDs from Redis `vector:*` keys or analysis cache index.
3. Recompute vectors with `FeatureExtractor.flatten()` order and update meta `vector_layout`.
4. Rebuild FAISS index from migrated vectors and verify dimension consistency.
5. Run similarity regression tests and compare score distributions.
