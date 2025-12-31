# DedupCAD Vision Vector ID Alignment Review (2025-12-31)

## Scope

- Verify whether dedupcad-vision registers vectors to cad-ml-platform using `file_hash` IDs.
- Identify alignment risks for `/api/compare` fallback which expects `candidate_hash` to match vector ID.

## Findings

- L3 compare path uses candidate hash as the lookup key:
  - `src/caddedup_vision/ml/client.py` → `compare_features(query_features, candidate_hash)`
  - It calls `/api/v1/vectors/search`, then falls back to `/api/compare` with `candidate_hash`.
- The only explicit registration path to cad-ml-platform is:
  - `src/caddedup_vision/integrations/routes.py` → `/integrations/vectors/register`
  - This endpoint accepts `doc_id` from the caller and forwards it to `/api/v1/vectors/register`.
- There is **no automatic mapping** in dedupcad-vision that guarantees `doc_id == file_hash`.

## Risk

- If vectors are registered with non-hash IDs (e.g., UUIDs), `/api/compare` and vector search will not find candidates by `candidate_hash`.

## Recommendation

- Ensure dedupcad-vision registers vectors with `doc_id = file_hash` when targeting cad-ml-platform for L3 comparison, or maintain a hash→id mapping layer.
