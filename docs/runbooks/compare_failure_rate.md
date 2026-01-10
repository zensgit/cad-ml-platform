# Runbook: Compare Request Failure Rate High

## Summary

Alert fires when `/api/compare` failure ratio stays above 30% with sustained traffic.

## Impact

- L3 fallback match quality degrades.
- Downstream dedup classification may rely only on L2/L1 signals.

## Signals

- `compare_requests_total` by status
- `compare_requests_total{status!="success"}` rate
- `compare_requests_total{status="not_found"}` rate

## Likely Causes

- Candidate hashes are not registered as vector IDs.
- Dimension mismatches between stored vectors and query features.
- Upstream vector store missing recent candidates.

## Triage

1. Check status breakdown:
   - `sum(rate(compare_requests_total[5m])) by (status)`
2. If `not_found` dominates, verify vector registration IDs match file hashes.
3. If `dimension_mismatch` dominates, confirm feature versions and layouts are aligned.
4. Inspect vector store health (`/health/extended`) and recent vector registrations.

## Mitigations

- Ensure dedupcad-vision registers `file_hash` as vector ID (or alias).
- Re-run vector registration for recent assets if cache missed.
- Roll back recent feature-vector layout changes if mismatch spikes.

## Recovery Verification

- Failure ratio falls below 30% for 10 minutes.
- Compare success rate returns to baseline.
