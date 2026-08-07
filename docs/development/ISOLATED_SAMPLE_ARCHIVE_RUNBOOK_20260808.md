# Isolated Sample / Archive Runbook

**Date**: 2026-08-08  
**Authority**: `PRODUCT_STRATEGY.md` §8.3 / §8.4; `CAD_REUSE_WORKBENCH_90_DAY_PLAN_20260807.md` §7–§8  
**Design-lock**: `L3_REVIEW_REUSE_WORKBENCH_DESIGNLOCK_20260808.md`  
**Status**: operator checklist (engineering + pilot ops)

---

## 1. Purpose

Define the **minimum isolation** rules for any customer or synthetic archive used to exercise the ReviewReuse workbench before pilot gates pass. This is a runbook, not a revived cost-cap module (#545).

---

## 2. Isolation rules (must all hold)

| # | Rule | How to verify |
|---|---|---|
| I1 | No shared multi-tenant processing of customer sample drawings | Dedicated API key / tenant; no co-tenancy with production archives |
| I2 | No hosted LLM provider egress for sample processing | `ASSISTANT_HOSTED_PROVIDER_OPT_IN` unset/false; do not call hosted vision/LLM providers with sample bytes |
| I3 | Sample storage location documented | Local path or private volume only (see §3) |
| I4 | Retention period documented | Default: **30 days** from ingest unless owner shortens |
| I5 | Deletion procedure documented | §4 below |
| I6 | Pre-pilot ban | Customer drawings are **not** processed in production or shared-tenant environments until pilot gates pass |

---

## 3. Storage

Recommended layout (operator-local or private volume):

```text
data/isolated_samples/<partner_or_synthetic_id>/
  README.md          # lawful basis, contact, retention end date
  archive/           # drawings (DXF preferred; PDF if allowed)
  exports/           # EvidencePack JSON/Markdown + task_id notes
  logs/              # redacted operator notes only
```

- Prefer synthetic or lawfully licensed samples for Day 31–60 MVP.
- Do not commit customer drawings to git.
- `source_content_sha256` in EvidencePack is the content integrity handle; keep a local mapping of sha → file name **off-repo**.

---

## 4. Deletion procedure

1. Cancel open tasks for the tenant (`POST .../tasks/{id}/cancel`).
2. Delete `data/isolated_samples/<id>/` (or equivalent volume path).
3. Restart API process if using in-memory MVP store (clears residual task objects).
4. Confirm no copies in logs, CI artifacts, or shared object storage.
5. Record deletion date + operator initials in the partner contact tracker (not in git).

---

## 5. End-to-end isolated archive exercise (MVP)

### 5.1 Prerequisites

```bash
export ENVIRONMENT=development
export API_KEY=test   # local only; never production "test"
# Decision sink remains OFF for default exercise:
unset REVIEW_REUSE_DECISIONS_ENABLED
# Optional pilot only:
# export REVIEW_REUSE_DECISIONS_ENABLED=true
uvicorn src.main:app --host 127.0.0.1 --port 8000
```

### 5.2 Create task (curl)

```bash
curl -sS -X POST "http://127.0.0.1:8000/api/v1/review-reuse/tasks" \
  -H "X-API-Key: test" \
  -F "file=@data/isolated_samples/synthetic_a/archive/sample.dxf" \
  -F "idempotency_key=isolated-demo-001" | tee exports/task.json
```

Offline / no-tool path yields one `insufficient_evidence` candidate with `tool_unavailable` — still a valid EvidencePack export.

### 5.3 Export EvidencePack

```bash
TASK_ID=$(python -c "import json;print(json.load(open('exports/task.json'))['task_id'])")
curl -sS -H "X-API-Key: test" \
  "http://127.0.0.1:8000/api/v1/review-reuse/tasks/${TASK_ID}/evidence-pack" \
  -o exports/evidence.json
curl -sS -H "X-API-Key: test" \
  "http://127.0.0.1:8000/api/v1/review-reuse/tasks/${TASK_ID}/evidence-pack?format=markdown" \
  -o exports/evidence.md
curl -sS -H "X-API-Key: test" \
  "http://127.0.0.1:8000/api/v1/review-reuse/tasks/${TASK_ID}/events" \
  -o exports/events.json
```

### 5.4 Optional human decision (owner enable only)

```bash
export REVIEW_REUSE_DECISIONS_ENABLED=true
# restart uvicorn
curl -sS -X POST \
  "http://127.0.0.1:8000/api/v1/review-reuse/tasks/${TASK_ID}/decision" \
  -H "X-API-Key: test" -H "Content-Type: application/json" \
  -d '{"state":"revise","reason_codes":["needs_dimension_check"],"reason_text":"pilot review","idempotency_key":"dec-001"}'
```

### 5.5 Service-level synthetic seed (tests / offline archive fixture)

Unit path uses `seed_candidates` on `ReviewReuseService.create_task` to simulate a recalled archive match without live dedup. See `tests/unit/test_review_reuse_workbench.py`.

---

## 6. Kill switch / rollback

| Action | Command / step |
|---|---|
| Disable decisions | unset or set `REVIEW_REUSE_DECISIONS_ENABLED=false`; restart |
| Stop accepting new tasks | take API out of load balancer / stop process |
| Clear MVP memory | restart process |
| Roll back deploy | redeploy previous image/SHA; no DB migration in MVP |

---

## 7. What this does **not** prove

- Not Track E model-promotion integrity (`eval_integrity_gate`).
- Not full §8.1 exit condition.
- Not production multi-tenant durability.
- Not customer commercial pilot success (Day 61–90).

It **does** prove the workbench "reproducible evaluator" for the **task / EvidencePack** family: a fresh operator can re-open stored task and evidence IDs and re-verify the exported pack.
