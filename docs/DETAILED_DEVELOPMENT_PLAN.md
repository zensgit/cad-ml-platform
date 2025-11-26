# Detailed Development Plan v2.3

Status: All phases (1A–6) complete; ≥68 tests passing.

## Follow-ups Priority
- P1: Auto-recovery persistence; expose next_recovery_eta
- P2: Preview percentiles; distribution breakdown
- P3: Ops issue templates; CI hardening

## Performance Baselines
- v4 p95: Temporary ≤4× v3; Target ≤2× v3 (lock-in 2025-Q1)
- Cache hit: Temporary ≥70%; Target ≥85% (30-day production)
- Recovery: Temporary ≤5min; Target ≤2min (after tuning)

## Changelog (v1.0 → v2.3)
- v4 features real (surface_count, shape_entropy) with latency metric
- Degraded mode instrumentation + auto-recovery loop
- Vector migration + preview stats (avg/median/warnings)
- Model reload security + opcode modes + audit endpoint
- Cache apply/rollback/prewarm endpoints
- Observability assets: metrics, alerts, Grafana dashboard
- CI stress workflow + scripts; docs/report/summary

## Acceptance Criteria per Phase
- Phase 1A: v4 math, 24 dims, ≥12 tests
- Phase 1B: Unified reload errors; preview stats extended
- Phase 2: Opcode modes; audit endpoint; metrics
- Phase 3: Auto-recovery; restored events metric
- Phase 4: Cache apply/rollback; prewarm; metrics
- Phase 5: Alerts + dashboard; metrics consistency
- Phase 6: Stress + memory/GC checks; ≥15 integration tests

## Status Footer
✅ All phases complete | 68+ tests passed

