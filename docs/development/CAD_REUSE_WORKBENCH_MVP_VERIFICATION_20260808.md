# CAD Reuse Workbench MVP — Verification

**Date**: 2026-08-08  
**Branch**: `l3-review-reuse-workbench-mvp-20260808`  
**PR**: https://github.com/zensgit/cad-ml-platform/pull/547  
**Plan**: `CAD_REUSE_WORKBENCH_MVP_DEVELOPMENT_PLAN_20260808.md`  
**Design-lock**: `L3_REVIEW_REUSE_WORKBENCH_DESIGNLOCK_20260808.md`  
**90-day plan**: `CAD_REUSE_WORKBENCH_90_DAY_PLAN_20260807.md`

---

## 1. Acceptance map (engineering MVP)

| Plan acceptance (§8 / design-lock §9) | Evidence |
|---|---|
| Design-lock binds §3.3 minimum fields | `docs/development/L3_REVIEW_REUSE_WORKBENCH_DESIGNLOCK_20260808.md` |
| API under `/api/v1/review-reuse/*` | `src/api/v1/review_reuse.py` + `src/api/__init__.py` + OpenAPI snapshot |
| Task events closed set | `TaskEventType` + unit assert on create path |
| EvidencePack JSON + Markdown | `evidence.py` + API `format=markdown` test + consumer smoke |
| Decision default-off | `REVIEW_REUSE_DECISIONS_ENABLED`; unit + API 403 + decision_gate.log |
| Tenant isolation | unit `test_tenant_isolation` + API different keys |
| Isolated-archive / synthetic e2e | seed candidates unit test + service smoke + runbook |
| No retrain / eval_integrity touch | `boundary_check.log` — no such paths in PR diff |
| Isolated sample runbook | `ISOLATED_SAMPLE_ARCHIVE_RUNBOOK_20260808.md` |
| Workflow authored | `.grok/workflows/cad-reuse-workbench-90d.rhai` |

**Not claimed:** owner design-lock ratification; customer pilot commercial next step; Track E model metrics; Day 61–90 measured pilot.

---

## 2. Commands run (real entry points)

```bash
# OpenAPI + route uniqueness (required for core-fast-gate) + workbench
pytest tests/contract/test_openapi_operation_ids.py \
       tests/contract/test_openapi_schema_snapshot.py \
       tests/unit/test_api_route_uniqueness.py \
       tests/unit/test_review_reuse_workbench.py \
       tests/unit/test_review_reuse_api.py -v

# Workbench suites (repeat)
pytest tests/unit/test_review_reuse_workbench.py tests/unit/test_review_reuse_api.py -q
```

### Results (Python 3.11.15, local)

| Run | Result |
|---|---|
| OpenAPI + uniqueness + workbench (21 tests) | **21 passed** in 2.14s |
| Workbench only re-run (16 tests) | **16 passed** in 1.69s |

Excerpt:

```text
tests/contract/test_openapi_schema_snapshot.py::test_openapi_schema_matches_snapshot PASSED
tests/unit/test_review_reuse_workbench.py::... PASSED (10)
tests/unit/test_review_reuse_api.py::... PASSED (6)
======================== 21 passed, 7 warnings in 2.14s ========================
................                                                         [100%]
16 passed, 7 warnings in 1.69s
```

OpenAPI snapshot refreshed via:

```bash
make openapi-snapshot-update
# OpenAPI snapshot written: config/openapi_schema_snapshot.json
# paths=199 operations=205  (includes /api/v1/review-reuse/*)
```

---

## 3. Consumer smokes (shipped service / mounted app)

### 3.1 EvidencePack (service entry)

Fresh `ReviewReuseService` + seed candidate → JSON has `task_id`, `trace_id`, geometric+semantic scores, verification, calibration version, `human_decision.allowed_actions` = reuse/revise/new; Markdown names task.

```text
STATUS=PASS
candidate scores: {'geometric': 0.88, 'semantic': 0.61}
calibration: {'version': 'workbench-mvp-0'}
allowed_actions: ['reuse', 'revise', 'new']
```

### 3.2 Decision gate

```text
disabled_path: code= decisions_disabled OK
enabled_path: status= decided decision= revise OK
http_disabled: 403 OK {'code': 'decisions_disabled', ...}
STATUS=PASS
```

---

## 4. Invariant / boundary checks

| ID | Check | Result |
|---|---|---|
| R1 | Decision default-off | PASS |
| R2 | No feedback training path | PASS (no feedback JSONL in workbench) |
| R3 | Tenant isolation | PASS |
| R4 | §3.3 fields present | PASS |
| R5 | No eval_integrity_gate edit | PASS (absent from `git diff origin/main...HEAD`) |
| R6 | No model-release metrics | PASS |
| R7 | No cost_cap revive | PASS (`src/core/assistant/cost_cap.py` absent) |
| R8 | No release authority | PASS (human POST only) |

PR path set (after OpenAPI fix): workbench sources, docs, tests, workflow, `.env.example`, `config/openapi_schema_snapshot.json`.

---

## 5. Residuals (explicit non-goals for this engineering tranche)

| Item | Status |
|---|---|
| Owner ratify design-lock | open (PROPOSED) |
| Enable `REVIEW_REUSE_DECISIONS_ENABLED` in pilot | owner-only |
| E1 / SEAL / identity post-merge audits | residual preserve |
| Customer Track C (contacts / samples / commercial) | residual |
| Live dedup adapter / durable store | follow-up eng |
| Day 61–90 measured pilot package | out of scope |

---

## 6. Workflow smoke

```text
workflow validate_only: cad-reuse-workbench-90d (args track=r-mvp) → metadata+compile path OK
```

---

## 7. PR / merge posture

| Item | Value |
|---|---|
| PR | https://github.com/zensgit/cad-ml-platform/pull/547 |
| Self-merge | **No** — wait for required CI green after OpenAPI snapshot commit |
| First core-fast-gate failure | OpenAPI snapshot missing review-reuse routes — **fixed** by snapshot update |

---

## 8. Sign-off

| Role | Action |
|---|---|
| Engineering | MVP code + tests + docs + OpenAPI snapshot on branch |
| CI required checks | re-check after snapshot push; do not self-merge on red |
| Owner | ratify design-lock; enable decisions only for pilot |
