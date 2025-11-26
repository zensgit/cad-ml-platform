---
name: Cache Rollback Request
about: Request rollback of cache settings applied via runtime controls
title: "Cache Rollback: <date/time>"
labels: ops, cache
assignees: 
---

## Current State
- Applied capacity:
- Applied ttl_seconds:
- Applied_at:
- can_rollback_until:

## Reason for Rollback

## Validation
- Recent hit ratio:
- Eviction rate:
- Errors observed:

## Action
- [ ] Execute rollback via `/api/v1/health/features/cache/rollback`
- [ ] Confirm metrics (`feature_cache_prewarm_total`, hit ratio)
- [ ] Document outcome

