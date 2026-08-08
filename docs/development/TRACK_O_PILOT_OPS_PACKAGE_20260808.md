# Track O — Pilot Operations Package

**Date**: 2026-08-08  
**Authority**: 90-day plan §8–§9 Track O; isolated-sample runbook  
**Status**: Engineering ops package (docs); complements runtime MVP

**Operator checklist**: step-by-step pilot posture (isolated samples, env flags matrix including `LIVE_DEDUP` / `STORE=filesystem` / `DECISIONS` / `REQUIRE_VALIDATED_REVIEWER`, audit-export, isolated-archive script, kill switch/rollback, residual design-lock + Track C) is in  
[`CAD_REUSE_WORKBENCH_PILOT_CHECKLIST_20260808.md`](./CAD_REUSE_WORKBENCH_PILOT_CHECKLIST_20260808.md).  
Decision remains default-off; R2 HOLD; Track C customer work is **not** claimed done here.

---

## 1. Purpose

Make pilot operation auditable before Day 61–90 measured pilot. This package is **report/export first**; it does not revive a silent cost_cap module.

---

## 2. Package inventory

| Item | Location / procedure |
|---|---|
| **Pilot operator checklist** | `CAD_REUSE_WORKBENCH_PILOT_CHECKLIST_20260808.md` |
| Isolated sample rules | `ISOLATED_SAMPLE_ARCHIVE_RUNBOOK_20260808.md` |
| Deployment | Private deploy of API image with `ENVIRONMENT` production posture + real `API_KEY` / `ADMIN_TOKEN` |
| Kill switch (decisions) | `REVIEW_REUSE_DECISIONS_ENABLED=false` (default); restart process |
| Kill switch (traffic) | Remove from LB / stop process |
| Backup | Export EvidencePack JSON + events (+ audit-export when available) per task to offline volume before changes |
| Rollback | Redeploy previous image SHA; restart clears MVP in-memory store |
| Audit export | `GET .../evidence-pack` (+ events); quarantine bundle `GET .../audit-export` when runtime present — see pilot checklist §3 |
| Isolated-archive script | `scripts/review_reuse_isolated_archive_run.py` — see pilot checklist §4 |
| Retention / deletion | Runbook §3–§4 (default 30 days; delete path documented) |
| Provider egress | SEAL: hosted AI off unless owner opt-in; samples never via hosted LLM |
| External AI cost | **If** owner enables hosted AI under §8.3, document budget owner + kill path; **do not** revive `cost_cap.py` without separate ratification |

---

## 3. Pilot metrics (review-workflow family only)

Collect from task ledger / human labels — **not** Track E model-release metrics:

| Metric | Source |
|---|---|
| task count | list tasks |
| top-5 candidate usefulness | human label on EvidencePack |
| accepted reuse | decisions with state=reuse |
| human false-duplicate labels | reason_codes / pilot spreadsheet |
| human missed-reuse labels | pilot spreadsheet |
| median review time | decision.ts − evidence_pack_ready event |
| reviewer coverage | distinct reviewer_id |
| insufficient_evidence rate | candidate state counts |

Optional future: execute-plan PR 4 metrics export endpoint.

---

## 4. Day-90 commercial next step (owner)

Record one of: paid pilot · contractual commitment · written pause/fold decision.  
**Not** an engineering deliverable.

---

## 5. Acceptance

| Criterion | Status |
|---|---|
| Runbook isolation complete | Done (sibling doc) |
| Kill/rollback/export/retention written | Done (this doc) |
| Metrics list defined | Done |
| Automated metrics export | Residual eng (PR 4) |
| Cost_cap module | Explicitly not shipped |

---

## 6. Operator quick path

```bash
# decisions off (safe default)
unset REVIEW_REUSE_DECISIONS_ENABLED
# pilot only
export REVIEW_REUSE_DECISIONS_ENABLED=true
# export
curl -H "X-API-Key: $API_KEY" \
  "$BASE/api/v1/review-reuse/tasks/$TASK_ID/evidence-pack" -o evidence.json
```
