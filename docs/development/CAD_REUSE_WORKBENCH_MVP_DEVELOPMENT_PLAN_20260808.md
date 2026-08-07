# CAD Reuse Workbench MVP — Development Plan

**Date**: 2026-08-08  
**Branch**: `l3-review-reuse-workbench-mvp-20260808`  
**Base**: `origin/main@7c9ae483` (post-#546)  
**Authority**: `CAD_REUSE_WORKBENCH_90_DAY_PLAN_20260807.md`, `PRODUCT_STRATEGY.md` §3.3 / §8.2  
**Design-lock**: `L3_REVIEW_REUSE_WORKBENCH_DESIGNLOCK_20260808.md`  
**Verification**: `CAD_REUSE_WORKBENCH_MVP_VERIFICATION_20260808.md`

---

## 1. Goal

Ship the **Day 31–60 engineering MVP** of the ReviewReuse workbench:

- design-lock for L3 objects;
- runtime task API under `/api/v1/review-reuse/*`;
- EvidencePack export (JSON + Markdown);
- human decision ledger **default-off**;
- isolated-sample runbook;
- unit + API tests;
- orchestration workflow for future full-plan runs.

Customer Track C (10 contacts, pilot commercial) remains residual.

---

## 2. Scope

### In

| Area | Deliverable |
|---|---|
| Domain | `src/core/review_reuse/{models,store,evidence,service}.py` |
| API | `src/api/v1/review_reuse.py` + register in `src/api/__init__.py` |
| Config | `REVIEW_REUSE_DECISIONS_ENABLED` in `.env.example` |
| Docs | design-lock, isolated-sample runbook, this plan, verification |
| Tests | `tests/unit/test_review_reuse_workbench.py`, `test_review_reuse_api.py` |
| Workflow | `.grok/workflows/cad-reuse-workbench-90d.rhai` |

### Out

- Live multi-process durable store
- Canonical PLM write-back
- Retrain unlock / `eval_integrity_gate` changes
- Track E model-run metrics
- Hosted LLM sample processing
- Silent `cost_cap` revival (#545)

---

## 3. Implementation sequence (executed)

1. **Design-lock** — bind §3.3 fields, API, enablement, quarantine, residuals.
2. **Domain service** — task lifecycle, events, EvidencePack, decision gate.
3. **HTTP API** — multipart create, list/get/cancel, events, evidence-pack, decision.
4. **Router registration** — after `dedup`, prefix `/review-reuse`.
5. **Isolated sample runbook** — storage, deletion, curl exercise, kill switch.
6. **Tests** — offline insufficient_evidence, seeded archive, tenant isolation, decision default-off/on, API routes.
7. **Workflow** — multi-phase agent orchestration for plan tracks (R/E/S/C/O).
8. **Verification MD** — evidence of tests + acceptance map.

---

## 4. API contract (shipped)

| Method | Path |
|---|---|
| POST | `/api/v1/review-reuse/tasks` |
| GET | `/api/v1/review-reuse/tasks` |
| GET | `/api/v1/review-reuse/tasks/{task_id}` |
| POST | `/api/v1/review-reuse/tasks/{task_id}/cancel` |
| GET | `/api/v1/review-reuse/tasks/{task_id}/events` |
| GET | `/api/v1/review-reuse/tasks/{task_id}/evidence-pack` |
| POST | `/api/v1/review-reuse/tasks/{task_id}/decision` |

Decision: **403** unless `REVIEW_REUSE_DECISIONS_ENABLED` is truthy.

---

## 5. Risk and mitigations

| Risk | Mitigation |
|---|---|
| Decision sink open by default | Env default-off; tests assert 403 |
| Training contamination | No feedback.py JSONL path; quarantine stated in design-lock |
| Cross-tenant leak | Store keyed by tenant; API tests with distinct keys |
| Clock / residual board confusion | Explicit non-goal; no #543 revive |
| L3 without owner ratify | Design-lock status PROPOSED; pilot enable separate |

---

## 6. Follow-ups (not this PR)

1. Live dedup2d adapter for recall/precision (replace offline seed).
2. Durable store for multi-worker pilot.
3. JWT-validated reviewer identity for pilot.
4. Customer Track C discovery residual.
5. Owner ratification of design-lock + decision enable for pilot only.
