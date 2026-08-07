# CAD Reuse Workbench MVP — Verification

**Date**: 2026-08-08  
**Branch**: `l3-review-reuse-workbench-mvp-20260808`  
**Plan**: `CAD_REUSE_WORKBENCH_MVP_DEVELOPMENT_PLAN_20260808.md`  
**Design-lock**: `L3_REVIEW_REUSE_WORKBENCH_DESIGNLOCK_20260808.md`  
**90-day plan**: `CAD_REUSE_WORKBENCH_90_DAY_PLAN_20260807.md`

---

## 1. Acceptance map (engineering MVP)

| Plan acceptance (§8 / design-lock §9) | Evidence |
|---|---|
| Design-lock binds §3.3 minimum fields | `docs/development/L3_REVIEW_REUSE_WORKBENCH_DESIGNLOCK_20260808.md` |
| API under `/api/v1/review-reuse/*` | `src/api/v1/review_reuse.py` + `src/api/__init__.py` |
| Task events closed set | `TaskEventType` + unit assert on create path |
| EvidencePack JSON + Markdown | `evidence.py` + API `format=markdown` test |
| Decision default-off | `REVIEW_REUSE_DECISIONS_ENABLED`; unit + API 403 |
| Tenant isolation | unit `test_tenant_isolation` + API different keys |
| Isolated-archive / synthetic e2e | seed candidates unit test + runbook curl path |
| No retrain / eval_integrity touch | git path review: only `review_reuse` + docs/tests/workflow |
| Isolated sample runbook | `ISOLATED_SAMPLE_ARCHIVE_RUNBOOK_20260808.md` |
| Workflow authored | `.grok/workflows/cad-reuse-workbench-90d.rhai` |

**Not claimed:** owner design-lock ratification; customer pilot commercial next step; Track E model metrics.

---

## 2. Commands run

```bash
# Unit (domain)
pytest tests/unit/test_review_reuse_workbench.py -v

# Unit (API)
pytest tests/unit/test_review_reuse_api.py -v

# Combined
pytest tests/unit/test_review_reuse_workbench.py tests/unit/test_review_reuse_api.py -v
```

Local run (2026-08-08, Python 3.11.15):

```text
pytest tests/unit/test_review_reuse_workbench.py tests/unit/test_review_reuse_api.py -v
======================== 16 passed, 7 warnings in 1.94s ========================
```

| Suite | Result |
|---|---|
| `test_review_reuse_workbench.py` (10 tests) | **PASSED** |
| `test_review_reuse_api.py` (6 tests) | **PASSED** |

---

## 3. Invariant checks

| ID | Check | How |
|---|---|---|
| R1 | Decision default-off | `decisions_enabled()` false without env; API 403 |
| R2 | No feedback training path | decision store is in-memory ReviewReuse only |
| R3 | Tenant isolation | cross-tenant get → not_found / 404 |
| R4 | §3.3 fields present | `TestEvidenceBuilder.test_section_33_minimum_fields` |
| R5 | No eval_integrity_gate edit | path set excludes `scripts/eval_integrity_gate*` |
| R6 | No model-release metrics | no Track E metric emission in service |
| R7 | No cost_cap revive | no `src/core/assistant/cost_cap.py` |
| R8 | No release authority | decision is human POST only; AI does not auto-decide |

---

## 4. Manual smoke (optional operator)

See runbook §5. With API up:

1. POST task with synthetic DXF bytes.
2. GET evidence-pack JSON and Markdown.
3. Confirm POST decision returns 403 with default env.
4. Enable flag, restart, POST `revise`, confirm `decided` and pack refresh.

---

## 5. Residual verification (owner / later)

| Item | Status |
|---|---|
| Owner ratify design-lock | open |
| E1 post-merge audit at exact head | residual |
| #537 / #538 baseline re-audit | residual (preserve) |
| Customer contacts / samples | residual Track C |
| Live dedup adapter | follow-up eng |
| Durable store | follow-up eng |

---

## 6. Workflow smoke

```text
workflow tool: validate_only on cad-reuse-workbench-90d.rhai
  with args: { "track": "r-mvp" }
```

Smoke-check proves metadata + compile + one canned path only — not live implementation.

---

## 7. Sign-off

| Role | Action |
|---|---|
| Engineering | MVP code + tests + docs shipped on branch |
| CI required checks | must be green before merge |
| Owner | ratify design-lock; enable decisions only for pilot |
