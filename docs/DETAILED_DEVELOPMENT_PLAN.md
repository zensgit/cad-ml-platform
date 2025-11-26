# CAD ML Platform – Detailed Development Plan (v2.3)

> Scope: Backend CAD FastAPI service, vector subsystem, degraded mode, model reload security, cache runtime controls, observability, CI, stress & stability.
> Target: Python 3.10+; mandatory type hints; metrics exported via `src/utils/analysis_metrics.py` `__all__`.

## Guiding Principles
- Minimal diffs per phase (≤300 core LOC, ≤5 new files).
- Security and observability first; performance tuned with metrics evidence.
- Explicit feature versioning; no silent upgrades.
- Dual auth for sensitive ops (`X-API-Key` + `X-Admin-Token`).
- Measurable acceptance criteria per phase; tests precede production rollout.

## Phase 1A — v4 Real Features (Complete)
- Implement `surface_count` (priority: metadata→entities→facets→solids).
- Implement `shape_entropy` with Laplace smoothing; normalize to [0,1].
- Keep v4 vector length fixed at 24 (22 geometric + 2 semantic).
- Record `feature_extraction_latency_seconds{version="v4"}`.
- Tests: ≥12 scenarios (empty/edge/uniform/skew/diversity/concurrency/perf). Achieved: 53 cases.

Acceptance
- All v4 tests pass; bounds respected; perf p95 ≤ 10ms or ≤ 4× v3 (temporary threshold).
- No regressions in other suites.

## Phase 1B — Error Unification + Preview Stats (Complete)
- Unify model reload errors via `create_extended_error` across: `size_exceeded`, `magic_invalid`, `hash_mismatch`, `opcode_blocked`, `opcode_scan_error`, `rollback`.
- Enhance migration preview with `avg_delta`, `median_delta`, `warnings`.
- Tests: `test_model_reload_errors_structured.py`, `test_migration_preview_stats.py`.

Acceptance
- Endpoint returns consistent error structure (code, stage, message, context, timestamp).
- Preview stats present and correct for diverse inputs.

## Phase 2 — Opcode Modes (Complete)
- Add `MODEL_OPCODE_MODE`: `blacklist` (default), `audit`, `whitelist`.
- Track `_OPCODE_AUDIT_SET`, `_OPCODE_AUDIT_COUNT`; expose `GET /api/v1/model/opcode-audit`.
- Metrics: `model_opcode_audit_total`, `model_opcode_whitelist_violations_total`.
- Tests: mode switching, audit persistence semantics, whitelist enforcement.

Acceptance
- Modes behave per spec; metrics increment appropriately; audit endpoint returns set+counts.

## Phase 3 — Faiss Auto-Recovery (Complete)
- Background loop with interval/backoff and `_FAISS_RECOVERY_LOCK`.
- Manual override endpoint; restoration increments `similarity_degraded_total{event="restored"}`.
- Metric: `faiss_recovery_attempts_total{result}`; duration gauge `faiss_degraded_duration_seconds`.
- Tests: recovery attempts, duration calculation, manual override coordination.

Acceptance
- Health shows degraded state/history; duration updates; restored events recorded.

## Phase 4 — Cache Apply/Rollback + Prewarm (Complete)
- Endpoints: `apply`, `rollback`, `prewarm`; 5‑minute rollback window; snapshot persistence.
- Prewarm priorities: recent hits, recent references, low‑dim high‑freq.
- Metrics: `feature_cache_prewarm_total{result}`.
- Tests: window enforcement, snapshot persistence, prewarm order.

Acceptance
- Safe apply with rollback window; prewarm results logged and metered; error paths produce structured errors.

## Phase 5 — Docs & Observability (Complete)
- Prometheus alerts: `prometheus/rules/cad_ml_phase5_alerts.yaml`.
- Grafana dashboard: `grafana/dashboards/observability.json`.
- Metrics consistency script: `scripts/metrics_consistency_check.py`.
- README updates: v4 math, preview stats, opcode modes, recovery, cache endpoints.

Acceptance
- CI verifies rules (promtool if available) and dashboard JSON; metrics script passes.

## Phase 6 — Stress & Stability (Complete)
- Scripts: `stress_concurrency_reload.py`, `stress_memory_gc_check.py`, `stress_degradation_flapping.py`.
- Integration tests: `tests/integration/test_stress_stability.py` (15 tests).
- CI: `.github/workflows/stress-tests.yml` matrix 3.10/3.11; artifacts and summary.

Acceptance
- ≥68 tests passing; stress scripts importable; CI summary generated.

## Deployment Plan
- Stage: Enable Prometheus scrape + Grafana dashboard; validate alerts with synthetic events.
- Canary: Roll out to small traffic slice; monitor degraded/restored rates, v4 latency p95, opcode violations, cache apply/rollback usage.
- Rollback: Use cache snapshot rollback; model reload guarded by hashes/opcodes; degraded mode remains safe fallback.

## Risk & Mitigations
- v4 perf variance: maintain absolute cap 10ms and ratio cap 2× once stabilized; monitor via dashboard.
- Auto-recovery flapping: use backoff/jitter; manual override flag to avoid race.
- Opcode migration: start in `audit`, switch to `whitelist` after observing production opcodes.
- Cache apply safety: refuse new apply during active rollback window.

## Metrics Checklist (must exist/export)
- Feature: `feature_extraction_latency_seconds{version}`
- Similarity: `vector_query_latency_seconds{backend}`, `vector_query_batch_latency_seconds{batch_size_range}`
- Migration: `vector_migrate_total{status}`, `vector_migrate_dimension_delta`
- Degraded: `similarity_degraded_total{event}`, `faiss_degraded_duration_seconds`, `faiss_recovery_attempts_total{result}`
- Opcode: `model_opcode_audit_total`, `model_opcode_whitelist_violations_total`
- Cache: `feature_cache_prewarm_total{result}`

## Testing Matrix
- Unit: v4 features (53), reload errors, opcode modes, degraded mode, cache endpoints, preview stats.
- Integration: stress stability (15), degradation flapping observation.
- Scripts: import and main() sanity in CI; optional local runs.

## CI/CD
- Path filters to limit runs; `workflow_dispatch` with scope selection.
- promtool rule validation (best effort); Grafana JSON parse check.
- Artifact retention 7 days; summary in Actions UI.

## Follow-ups (Optional Enhancements)

| Priority | Enhancement | Description |
|----------|-------------|-------------|
| **P1** | Auto-recovery persistence | Persist backoff state across restarts; expose next attempt ETA in health endpoint |
| **P2** | Preview percentiles | Add p50/p90/p99 to migration preview; tighten v4 perf threshold to 2× |
| **P3** | Ops issue templates | GitHub issue templates for degraded incidents, cache rollback requests |

## Performance Baseline

| Metric | Temporary Threshold | Target Threshold | Lock-in Date |
|--------|---------------------|------------------|--------------|
| v4 extraction p95 | ≤10ms or ≤4× v3 | ≤10ms or ≤2× v3 | 2025-Q1 review |
| Cache hit ratio | ≥70% | ≥85% | After 30-day production data |
| Degraded recovery | ≤5min auto | ≤2min auto | After backoff tuning |

## File References
- v4 features: `src/core/feature_extractor.py:1`
- Vectors & preview: `src/api/v1/vectors.py`
- Degraded & recovery: `src/core/similarity.py`, `src/api/v1/health.py`
- Model reload & opcode modes: `src/ml/classifier.py`, `src/api/v1/model.py`
- Cache controls: `src/core/feature_cache.py`, `src/api/v1/health.py`
- Metrics: `src/utils/analysis_metrics.py`
- Alerts/Dashboard: `prometheus/rules/cad_ml_phase5_alerts.yaml`, `grafana/dashboards/observability.json`
- CI: `.github/workflows/stress-tests.yml`

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| v2.3 | 2025-11-26 | All 6 phases complete; 68 tests passing; added priority table, performance baseline |
| v2.2 | 2025-11-25 | Phase 5-6 complete; stress tests and observability infrastructure |
| v2.1 | 2025-11-24 | Phase 3-4 complete; Faiss auto-recovery, cache apply/rollback |
| v2.0 | 2025-11-23 | Phase 1-2 complete; v4 features, opcode modes, error unification |
| v1.0 | 2025-11-22 | Initial plan draft |

---

**Status**: ✅ All phases complete | **Tests**: 68 passed | **Last updated**: 2025-11-26

