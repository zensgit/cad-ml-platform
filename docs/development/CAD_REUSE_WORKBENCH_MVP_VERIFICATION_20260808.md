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

## 5. Residuals

| Item | Status |
|---|---|
| Owner ratify design-lock | residual_human (PROPOSED) |
| Enable `REVIEW_REUSE_DECISIONS_ENABLED` in pilot | residual_human |
| E1 post-merge audit | `TRACK_E_E1_POST_MERGE_AUDIT_20260808.md` |
| SEAL / identity re-audit | `TRACK_S_SEAL_IDENTITY_BASELINE_AUDIT_20260808.md` |
| Track O pilot ops package | `TRACK_O_PILOT_OPS_PACKAGE_20260808.md` |
| System task board + execute design | `CAD_REUSE_WORKBENCH_TASK_BOARD_20260808.md`, `CAD_REUSE_WORKBENCH_SYSTEM_EXECUTE_DESIGN_20260808.md` |
| Customer Track C | residual_human |
| Live dedup adapter / metrics export | residual_eng — execute-plan PR3/PR4 **after** #547 (L3 WIP=1) |
| Day 61–90 measured pilot package | residual_human / eng support |

---

## 6. Workflow smoke + system

```text
cad-reuse-workbench-90d   validate_only track=r-mvp → OK
cad-reuse-workbench-dev   validate_only mode=gap|full → OK
cad-reuse-workbench-system validate_only phase=all → OK
```

### Live run: `cad-reuse-workbench-dev` mode=verify (2026-08-08)

| Field | Result |
|---|---|
| overall `ok` | **true** |
| gap_ok_areas | **4/4** |
| tests_pass | **true** (16/16 review-reuse unit; + OpenAPI/route uniqueness 3/3) |
| verify_ok | **true** |
| missing | `[]` |
| report | `scratch/workbench_dev_report.md` (workflow run artifact) |

Workflow next_actions (owner / residual, not eng self-complete): design-lock ratification; pilot-only decision enable; Track C human residual; leave eval_integrity/cost_cap/retrain out of Track R.

### Live run: `cad-reuse-workbench-system` phase=all (2026-08-08)

| Field | Result |
|---|---|
| overall `ok` | **true** |
| phase | all |
| inventory | Track R MVP + residual audits/ops/system **engineering-done** on #547 |
| residual_eng open | R10 live dedup (PR3), O3 metrics (PR4) — **blocked by L3 WIP=1** |
| residual_human | R11 design-lock ratify, R12 decision enable, Track C C1–C5 |
| execute_plan_design | `CAD_REUSE_WORKBENCH_SYSTEM_EXECUTE_DESIGN_20260808.md` |
| report | `scratch/workbench_system_report.md` |

System blockers (expected): do not open second L3 runtime PR while #547 open; no self-merge on red CI; PR1+PR2 docs already on #547.

Scheduled task: `cad-reuse-workbench-gap-check` (weekdays 09:30 Asia/Shanghai).

execute-plan design: `CAD_REUSE_WORKBENCH_SYSTEM_EXECUTE_DESIGN_20260808.md`  
PR1+PR2 content landed on this PR; PR3+PR4 gated on L3 slot after #547 merges.

---

## 7. PR / merge posture

| Item | Value |
|---|---|
| PR | https://github.com/zensgit/cad-ml-platform/pull/547 |
| Self-merge | **No** — wait for required CI green |
| OpenAPI snapshot | refreshed for `/api/v1/review-reuse/*` |

---

## 8. Sign-off

| Role | Action |
|---|---|
| Engineering | MVP + residual audits/ops + system (workflows/board/execute design) on branch |
| CI required checks | green before merge |
| Owner | ratify design-lock; pilot decision enable; Track C |
