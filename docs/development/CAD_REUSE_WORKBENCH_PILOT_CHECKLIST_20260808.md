# CAD Reuse Workbench — Pilot Operator Checklist

**Date**: 2026-08-08  
**Branch intent**: `docs/workbench-pilot-checklist-20260808` (docs only)  
**Authority**: `PRODUCT_STRATEGY.md` §8.2–§8.4; 90-day plan Track O; design-lock R1–R8  
**Status**: Operator checklist for isolated pilot ops — **not** a claim that Track C or design-lock ratification is complete  

**Related docs**

| Doc | Role |
|---|---|
| [`ISOLATED_SAMPLE_ARCHIVE_RUNBOOK_20260808.md`](./ISOLATED_SAMPLE_ARCHIVE_RUNBOOK_20260808.md) | Isolation rules I1–I6, storage layout, deletion |
| [`TRACK_O_PILOT_OPS_PACKAGE_20260808.md`](./TRACK_O_PILOT_OPS_PACKAGE_20260808.md) | Ops inventory, metrics family, acceptance |
| [`CAD_REUSE_WORKBENCH_LIVE_DEDUP_DURABLE_STORE_20260808.md`](./CAD_REUSE_WORKBENCH_LIVE_DEDUP_DURABLE_STORE_20260808.md) | Live dedup + filesystem store defaults |
| [`CAD_REUSE_WORKBENCH_EXTERNAL_GATES_AUDIT_20260808.md`](./CAD_REUSE_WORKBENCH_EXTERNAL_GATES_AUDIT_20260808.md) | R2 HOLD + open external/human gates |
| [`L3_REVIEW_REUSE_WORKBENCH_DESIGNLOCK_20260808.md`](./L3_REVIEW_REUSE_WORKBENCH_DESIGNLOCK_20260808.md) | L3 design-lock (owner ratify residual) |
| [`CAD_REUSE_WORKBENCH_JWT_PILOT_RUNBOOK_20260808.md`](./CAD_REUSE_WORKBENCH_JWT_PILOT_RUNBOOK_20260808.md) | JWT pilot env matrix + store backup/cleanup |

---

## 0. Non-negotiable holds (read before enabling anything)

| Hold | Operator rule |
|---|---|
| **R1 — Decision default-off** | Leave `REVIEW_REUSE_DECISIONS_ENABLED` unset/false unless owner explicitly enables for a named pilot window. Decision POST must 403 when off. |
| **R2 HOLD — No training-path reuse** | Audit / EvidencePack exports are **quarantined** review artifacts. Do **not** copy them into training manifests, feedback JSONL, or `eval_integrity` promotion paths. |
| **R5 / R6** | Workbench does not replace `eval_integrity_gate` or Track E model-release metrics. Pilot metrics are `review_workflow` only. |
| **R7** | Do not revive `cost_cap.py` for sample budgets. |
| **R8 — AI has no release authority** | Only a human decision (when enabled) records reuse/revise/new. |
| **SEAL** | Hosted LLM/provider egress stays opt-in and **off** for sample drawings (`ASSISTANT_HOSTED_PROVIDER_OPT_IN` unset/false). |

This checklist does **not** complete Track C (contacts, sample conversations, named reviewer, measured pilot) and does **not** ratify the L3 design-lock. Those remain residual human gates (§7).

---

## 1. Isolated sample rules (must all hold)

Copy of the runbook contract — verify each row before any customer or synthetic archive touches a pilot API process.

| # | Rule | How to verify |
|---|---|---|
| I1 | No shared multi-tenant processing of customer sample drawings | Dedicated API key / tenant; no co-tenancy with production archives |
| I2 | No hosted LLM provider egress for sample processing | `ASSISTANT_HOSTED_PROVIDER_OPT_IN` unset/false; do not call hosted vision/LLM with sample bytes |
| I3 | Sample storage location documented | Local path or private volume only (layout below) |
| I4 | Retention period documented | Default **30 days** from ingest unless owner shortens |
| I5 | Deletion procedure documented | Runbook §4 + this checklist §6 |
| I6 | Pre-pilot ban | Customer drawings **not** in production or shared-tenant env until pilot gates pass |

### Recommended layout

```text
data/isolated_samples/<partner_or_synthetic_id>/
  README.md          # lawful basis, contact, retention end date
  archive/           # drawings (DXF preferred; PDF if allowed)
  exports/           # EvidencePack + audit bundle + task notes
  logs/              # redacted operator notes only
```

- Prefer synthetic or lawfully licensed samples for Day 31–60 MVP.
- Do **not** commit customer drawings to git.
- Keep sha256 → file name mapping **off-repo**.

Full procedure: [`ISOLATED_SAMPLE_ARCHIVE_RUNBOOK_20260808.md`](./ISOLATED_SAMPLE_ARCHIVE_RUNBOOK_20260808.md).

---

## 2. Env flags matrix (pilot posture)

All workbench flags are **default-off or memory-local** unless the operator intentionally enables them. Restart the API process after any change.

| Env var | Safe / default | Pilot optional | Behavior |
|---|---|---|---|
| `REVIEW_REUSE_DECISIONS_ENABLED` | **unset / false** | `true` (owner-only window) | When off: `POST .../decision` → 403 (`decisions_disabled`). **Must stay default-off** outside an explicit pilot. |
| `REVIEW_REUSE_LIVE_DEDUP` | **unset / false** | `true` (private vision only) | When off: offline path → `insufficient_evidence` / `tool_unavailable`. When on: private `DedupCadVisionClient.search_2d` (or injected hook). Failures fail closed. **No hosted LLM.** |
| `REVIEW_REUSE_STORE` | `memory` | `filesystem` | `memory`: process-local (lost on restart). `filesystem`: restart-safe single-node pilot under `REVIEW_REUSE_STORE_DIR`. |
| `REVIEW_REUSE_STORE_DIR` | `data/review_reuse_tasks` | private volume path | Used only when `REVIEW_REUSE_STORE=filesystem`. |
| `REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER` | legacy compatibility only | no pilot choice | Cannot weaken the mandatory validated tenant + reviewer rule. API-key fallbacks are always read/create-only. |

### Recommended postures

**A. Safe offline exercise (default)**

```bash
export ENVIRONMENT=development
export API_KEY=test   # local only — never production "test"
unset REVIEW_REUSE_DECISIONS_ENABLED
unset REVIEW_REUSE_LIVE_DEDUP
unset REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER
# optional durable local store:
export REVIEW_REUSE_STORE=filesystem
export REVIEW_REUSE_STORE_DIR=data/review_reuse_tasks
```

**B. Isolated pilot with durable store + private live dedup (still no decisions)**

```bash
export REVIEW_REUSE_STORE=filesystem
export REVIEW_REUSE_STORE_DIR=/var/lib/cad-ml/review_reuse_tasks   # private volume
export REVIEW_REUSE_LIVE_DEDUP=true
# DEDUPCAD_VISION_URL=...   # private on-prem vision only
unset REVIEW_REUSE_DECISIONS_ENABLED
unset REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER
```

**C. Named-reviewer decision window (owner enable only)**

```bash
export REVIEW_REUSE_DECISIONS_ENABLED=true
export INTEGRATION_AUTH_MODE=required
export INTEGRATION_JWT_SECRET=...
export INTEGRATION_JWT_AUDIENCE=...
export INTEGRATION_JWT_ISSUER=...
# A valid Bearer JWT must provide exact string sub + tenant_id claims.
# Restart API after env change
```

Document the enable window (who, why, start/end) offline — not in git with secrets.

See also `.env.example` ReviewReuse block and [`CAD_REUSE_WORKBENCH_LIVE_DEDUP_DURABLE_STORE_20260808.md`](./CAD_REUSE_WORKBENCH_LIVE_DEDUP_DURABLE_STORE_20260808.md).

---

## 3. Audit-export usage

Audit export is a **quarantined review bundle** (`export_kind=audit_quarantine`). It is **not** a training manifest (R2 HOLD).

### HTTP (preferred for pilot ledger)

```bash
export BASE="${BASE:-http://127.0.0.1:8000}"
export API_KEY="${API_KEY:-test}"

# Per-task EvidencePack (JSON + Markdown)
curl -sS -H "X-API-Key: $API_KEY" \
  "$BASE/api/v1/review-reuse/tasks/${TASK_ID}/evidence-pack" \
  -o exports/evidence.json
curl -sS -H "X-API-Key: $API_KEY" \
  "$BASE/api/v1/review-reuse/tasks/${TASK_ID}/evidence-pack?format=markdown" \
  -o exports/evidence.md

# Events ledger
curl -sS -H "X-API-Key: $API_KEY" \
  "$BASE/api/v1/review-reuse/tasks/${TASK_ID}/events" \
  -o exports/events.json

# Quarantined audit bundle (task + events + EvidencePack JSON/MD)
# Runtime surface: GET /api/v1/review-reuse/tasks/{id}/audit-export
curl -sS -H "X-API-Key: $API_KEY" \
  "$BASE/api/v1/review-reuse/tasks/${TASK_ID}/audit-export" \
  -o exports/audit_bundle.json
```

### Bundle contract (operator expectations)

| Field | Meaning |
|---|---|
| `schema_version` | e.g. `review-reuse-audit-bundle-v1` |
| `export_kind` | `audit_quarantine` — not training-readable |
| `task` | Full task snapshot |
| `events` | Append-only event list |
| `evidence_pack` | Strategy §3.3 field family (JSON) |
| `evidence_pack_markdown` | Reviewer-facing Markdown |

### What operators must **not** do with exports

- Do not feed audit bundles into training JSONL / feedback flywheel / model promotion gates.
- Do not store customer export paths in git or public object storage.
- Prefer offline volume under `data/isolated_samples/<id>/exports/`.

Backup rule (Track O): export EvidencePack + events (and audit bundle when available) **before** deploy changes or deletion.

---

## 4. Isolated-archive script command

Offline / CI-friendly exercise: create a `ReviewReuseTask` → write EvidencePack + audit bundle to disk. **Does not enable human decisions.**

Preferred operator shortcut (sets `ENVIRONMENT=development`, passes `--seed-similar`, does **not** enable `REVIEW_REUSE_DECISIONS_ENABLED`):

```bash
make review-reuse-isolated-archive
```

Equivalent direct invocation and optional variants:

```bash
# Synthetic bytes + seed-similar (same as the Make target)
python scripts/review_reuse_isolated_archive_run.py \
  --out data/isolated_samples/synthetic_run/exports \
  --seed-similar

# Optional: real file + synthetic similar candidate (offline archive fixture)
python scripts/review_reuse_isolated_archive_run.py \
  --file data/isolated_samples/synthetic_a/archive/sample.dxf \
  --seed-similar \
  --tenant isolated-sample \
  --idempotency-key isolated-archive-demo \
  --out data/isolated_samples/synthetic_a/exports
```

| Flag | Purpose |
|---|---|
| `--out` | Export directory (`task.json`, `evidence.json`, `evidence.md`, `audit_bundle.json`) |
| `--file` | Optional drawing; otherwise minimal synthetic DXF bytes |
| `--tenant` | Tenant id for isolation (default `isolated-sample`) |
| `--seed-similar` | Attach offline similar candidate without live vision |
| `--idempotency-key` | Create-task idempotency key |

Optional env while running the script:

```bash
export REVIEW_REUSE_STORE=filesystem
export REVIEW_REUSE_STORE_DIR=data/review_reuse_tasks
# Keep live dedup off for pure offline isolation demos:
unset REVIEW_REUSE_LIVE_DEDUP
# Script pops REVIEW_REUSE_DECISIONS_ENABLED — do not re-export it true in the same shell for the script path
```

Expected stdout includes `decisions=disabled` and paths under `--out`.

---

## 5. End-to-end HTTP exercise (curl)

### 5.1 Start API (safe defaults)

```bash
export ENVIRONMENT=development
export API_KEY=test
unset REVIEW_REUSE_DECISIONS_ENABLED
export REVIEW_REUSE_STORE=filesystem
uvicorn src.main:app --host 127.0.0.1 --port 8000
```

### 5.2 Create task

```bash
curl -sS -X POST "http://127.0.0.1:8000/api/v1/review-reuse/tasks" \
  -H "X-API-Key: test" \
  -F "file=@data/isolated_samples/synthetic_a/archive/sample.dxf" \
  -F "idempotency_key=isolated-demo-001" | tee exports/task.json

TASK_ID=$(python -c "import json;print(json.load(open('exports/task.json'))['task_id'])")
```

Offline / no-tool path may yield one `insufficient_evidence` candidate with `tool_unavailable` — still a valid EvidencePack export.

### 5.3 Export EvidencePack + audit

Use §3 commands with `$TASK_ID`.

### 5.4 Optional human decision (owner enable only)

This command is valid only after a separate owner authorization opens a named
decision window and the server has restarted in the full required-JWT posture
from §2.C. It is not part of the default isolated exercise.

```bash
export REVIEW_REUSE_DECISIONS_ENABLED=true
# restart uvicorn with INTEGRATION_AUTH_MODE=required and complete JWT config

# Values must come from the exact EvidencePack loaded by this reviewer.
TASK_REVISION="$(python -c 'import json; print(json.load(open("exports/evidence.json"))["task_revision"])')"
EVIDENCE_SHA="$(python -c 'import json; print(json.load(open("exports/evidence.json"))["evidence_pack_sha256"])')"
CANDIDATE_ID="$(python -c 'import json; print(json.load(open("exports/evidence.json"))["candidates"][0]["candidate_id"])')"
JWT='<validated reviewer bearer token>'

curl -sS -X POST \
  "http://127.0.0.1:8000/api/v1/review-reuse/tasks/${TASK_ID}/decision" \
  -H "X-API-Key: $API_KEY" \
  -H "Authorization: Bearer ${JWT}" \
  -H "Content-Type: application/json" \
  -d "{\"state\":\"revise\",\"candidate_id\":\"${CANDIDATE_ID}\",\"expected_revision\":${TASK_REVISION},\"evidence_pack_sha256\":\"${EVIDENCE_SHA}\",\"reason_codes\":[\"needs_modification\"],\"reason_text\":\"pilot review\",\"idempotency_key\":\"dec-001\"}"
```

An API-key-only request cannot submit a decision. With valid platform/JWT
authentication but decisions default-off, expect **403**
`decisions_disabled`.

---

## 6. Kill switch / rollback

| Action | Command / step |
|---|---|
| Disable decisions (primary kill switch) | `unset REVIEW_REUSE_DECISIONS_ENABLED` or set `=false`; **restart** process |
| Validated reviewer rule | Mandatory for every decision; the legacy env flag is not a bypass or kill switch |
| Disable live dedup | `unset REVIEW_REUSE_LIVE_DEDUP` or `=false`; restart |
| Stop accepting new tasks | Remove from load balancer / stop process |
| Clear memory store | Restart process (`REVIEW_REUSE_STORE=memory`) |
| Clear filesystem store | Stop process; delete or archive `REVIEW_REUSE_STORE_DIR` for the pilot tenant; restart |
| Sample deletion | Cancel open tasks → delete `data/isolated_samples/<id>/` → confirm no copies in logs/CI/shared storage → record deletion off-repo |
| Roll back deploy | Redeploy previous image/SHA; MVP has no multi-tenant DB migration for workbench |

**Safe default after any incident:** decisions off, live dedup off, hosted providers off, process stopped until owner re-approves posture B or C.

---

## 7. Residual human gates (not claimed done by this doc)

Engineering ops docs and runtime gates do **not** close these. Treat as **OPEN** unless owner records otherwise offline.

| Gate | Owner | Status to assume |
|---|---|---|
| **Design-lock ratify** (`L3_REVIEW_REUSE_WORKBENCH_DESIGNLOCK_*`) | Product owner | **OPEN** — AI has no release authority; owner pins head |
| **Pilot enable decisions** | Owner | **OPEN (default-off)** — explicit window only |
| **Track C: ≥10 qualified contacts** | Owner | **OPEN** — not claimed complete |
| **Track C: ≥2 lawful sample-data conversations** | Owner | **OPEN** — not claimed complete |
| **Track C: named reviewer + baseline metrics** | Owner | **OPEN** — not claimed complete |
| **Track C: isolated customer archive** | Owner + eng support | **OPEN** — needs lawful customer data |
| **Track C: measured pilot / commercial next step** | Owner | **OPEN** — Day 90 business gate |

Do **not** invent Track C evidence, flip decision default-on in code, or treat this checklist merge as design-lock ratification.

External gates audit: [`CAD_REUSE_WORKBENCH_EXTERNAL_GATES_AUDIT_20260808.md`](./CAD_REUSE_WORKBENCH_EXTERNAL_GATES_AUDIT_20260808.md).

---

## 8. Pre-flight checkbox (print / tick offline)

```text
[ ] I1–I6 isolation verified for this sample id
[ ] Hosted provider opt-in OFF
[ ] Decisions OFF (or owner-signed enable window documented)
[ ] Live dedup OFF unless private vision URL configured
[ ] STORE=filesystem on private volume for multi-restart pilot
[ ] REQUIRE_VALIDATED_REVIEWER ON whenever decisions ON
[ ] Audit export path writable under isolated_samples/.../exports
[ ] Kill switch / rollback owner named
[ ] Design-lock still PROPOSED or separately ratified (do not self-claim)
[ ] Track C items still residual (do not self-claim)
```

---

## 9. What this checklist proves vs does not

| Proves | Does not prove |
|---|---|
| Operator can run isolated task → EvidencePack / audit export under fail-closed defaults | Track E model promotion integrity |
| Kill switch and rollback steps are written and actionable | Production multi-tenant durability / Redis store |
| Env matrix keeps decision and live dedup default-off | Customer commercial pilot success |
| Residual human gates are explicitly listed open | Design-lock owner ratification |

It supports the workbench as a **reproducible evaluator** for the task / EvidencePack family under R1–R8 holds.
