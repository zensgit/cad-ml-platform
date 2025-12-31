# Runbook: Compare Not Found Dominant

## Summary

Alert fires when more than half of `/api/compare` requests return `not_found`.

## Impact

- Compare fallback requests cannot locate stored candidate vectors.
- L3 fallback yields empty or low-confidence scoring.

## Signals

- `compare_requests_total{status="not_found"}`
- `compare_requests_total` by status

## Likely Causes

- Candidate hash does not match vector ID in the store.
- Vectors are being pruned or not registered.
- Dedupe pipeline using stale IDs after cache refresh.

## Triage

1. Confirm vector registration path is active.
2. Validate candidate hashes against vector IDs.
3. Check vector store size (`/health/extended`) and prune activity.
4. Inspect recent ingest jobs and vector TTL settings.

## Mitigations

- Register alias vectors with `file_hash` IDs.
- Disable aggressive pruning or increase TTL while backfilling.
- Re-index vectors from source of truth if store is empty.

## Recovery Verification

- `not_found` ratio falls below 50% for 10 minutes.
- `/api/compare` success rate normalizes.
