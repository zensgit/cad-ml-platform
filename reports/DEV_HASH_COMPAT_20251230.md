# Hash Compatibility Assessment

- Date: 2025-12-30
- Scope: MD5/SHA1 → SHA256 changes in vision-related hashing.

## Summary
- MD5/SHA1 usage in `src/core/vision/*` was replaced with SHA256. This affects any cache keys, request IDs, or fingerprints derived from those hashes.

## Impact Areas
- **In-memory caches**: will experience cold starts because keys change.
- **Persistent stores**: any persisted IDs or keys derived from MD5 (e.g., feature flags, request IDs, knowledge base entry IDs) will no longer match new SHA256-derived values.
- **External integrations**: if downstream systems expect a specific hash length/value, they will need to accept SHA256.

## Recommended Actions
- Flush or rebuild caches/indexes that store MD5-based keys (Faiss/metadata caches and any on-disk artifacts tied to MD5-derived IDs).
- If backwards compatibility is required, introduce dual-key lookups or a migration step that rewrites stored IDs.
- Confirm no client/API contract relies on exact hash string length; update docs or integrations if necessary.

## Files Touched (Representative)
- `src/core/vision/*` (multiple modules; MD5 → SHA256)
- `src/core/vision/deduplication.py` (HashAlgorithm options reduced; defaults SHA256)
- `src/core/vision/embedding.py` (MD5 removed from cryptographic hash generator)
