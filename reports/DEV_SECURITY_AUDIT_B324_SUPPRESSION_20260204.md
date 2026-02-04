# DEV_SECURITY_AUDIT_B324_SUPPRESSION_20260204

## Summary
- Marked MD5 usages that are non-security-critical with `# nosec B324` to satisfy bandit high-severity checks.
- Targeted cache keys, ETag compatibility paths, idempotency fingerprints, rollout bucketing, and artifact checksums.
- No behavioral changes to hashing outputs; only bandit annotations.

## Files Updated
- `src/core/caching/__init__.py`
- `src/core/cad/dwg/manager.py`
- `src/core/feature_toggles_enhanced/toggle.py`
- `src/core/idempotency/__init__.py`
- `src/core/service_mesh/discovery.py`
- `src/core/service_mesh/load_balancer.py`
- `src/core/storage/multipart.py`
- `src/core/storage/object_store.py`
- `src/ml/experiment/artifacts.py`
- `src/ml/serving/ab_testing.py`

## Verification
- Security Audit CI will re-run with the new annotations; no local test changes required.
