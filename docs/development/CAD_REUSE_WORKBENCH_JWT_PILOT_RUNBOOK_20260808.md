# CAD Reuse Workbench — JWT Pilot Runbook + Env Matrix

**Date**: 2026-08-08  
**Status**: Operator runbook (docs + ops tooling companion)  
**Authority**: design-lock (reviewer identity); R1 decision default-off; R2 HOLD  
**Does not**: ratify design-lock, enable production decisions, or complete Track C

**Related**

| Doc | Role |
|---|---|
| [`CAD_REUSE_WORKBENCH_PILOT_CHECKLIST_20260808.md`](./CAD_REUSE_WORKBENCH_PILOT_CHECKLIST_20260808.md) | Full pilot checklist |
| [`CAD_REUSE_WORKBENCH_LIVE_DEDUP_DURABLE_STORE_20260808.md`](./CAD_REUSE_WORKBENCH_LIVE_DEDUP_DURABLE_STORE_20260808.md) | Live dedup + store |
| `.env.example` | Integration JWT + ReviewReuse flags |

---

## 1. Goal

Pilot decisions must attribute **validated human subjects** (JWT `sub`), not API-key fallbacks (`ak-user-*`).

Identity is set only by integration auth middleware from a verified token:

- `request.state.user_id` / `request.state.auth_subject` ← token `sub`
- `request.state.tenant_id` ← token `tenant_id`

Workbench maps those into `reviewer_id` with `reviewer_validated=True`.  
If only `X-API-Key` is present, reviewer is `ak-user-<hash>` and **fails** when `REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER=true`.

---

## 2. Env matrix (copy for pilot window)

### 2.1 Identity / integration JWT

| Variable | Offline / CI | Pilot with decisions | Notes |
|---|---|---|---|
| `ENVIRONMENT` | `development` / `test` | `production` or staging equivalent | Production fail-closed identity rules apply outside dev/test |
| `API_KEY` / `API_KEYS` | local secret | real secret | Never ship `test` in production |
| `INTEGRATION_AUTH_MODE` | `disabled` or `optional` | **`required`** for pilot decisions | Middleware validates Bearer JWT |
| `INTEGRATION_JWT_SECRET` | optional | **required** when mode=required | Shared secret / key material |
| `INTEGRATION_JWT_AUDIENCE` | optional | **required** when mode=required | Aud claim |
| `INTEGRATION_JWT_ISSUER` | optional | **required** when mode=required | Iss claim |
| `INTEGRATION_JWT_ALG` | `HS256` | as issued | Match token |

Token **must** include claims: `sub`, `tenant_id`, `exp`, `iat` (see middleware).

### 2.2 ReviewReuse workbench

| Variable | Safe default | Pilot exercise | Pilot decisions (owner-approved only) |
|---|---|---|---|
| `REVIEW_REUSE_DECISIONS_ENABLED` | **off** | **off** | **on** for named window only |
| `REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER` | off | off or on | **on** when decisions on |
| `REVIEW_REUSE_LIVE_DEDUP` | off | optional on (private vision) | same |
| `REVIEW_REUSE_STORE` | `memory` | `filesystem` | `filesystem` |
| `REVIEW_REUSE_STORE_DIR` | `data/review_reuse_tasks` | private volume | private volume |
| `ASSISTANT_HOSTED_PROVIDER_OPT_IN` | off | **off** for samples | **off** for samples |

### 2.3 Recommended postures

**A — Offline exercise (default)**  
```bash
unset REVIEW_REUSE_DECISIONS_ENABLED
unset REVIEW_REUSE_LIVE_DEDUP
export REVIEW_REUSE_STORE=memory   # or filesystem for restart tests
export INTEGRATION_AUTH_MODE=disabled
# make review-reuse-isolated-archive
```

**B — Isolated pilot, evidence only (no decisions)**  
```bash
unset REVIEW_REUSE_DECISIONS_ENABLED
export REVIEW_REUSE_STORE=filesystem
export REVIEW_REUSE_STORE_DIR=/secure/pilot/review_reuse_tasks
export REVIEW_REUSE_LIVE_DEDUP=true   # only if private DedupCAD Vision is up
export INTEGRATION_AUTH_MODE=optional   # or required if all clients have JWT
```

**C — Owner-approved decision window**  
```bash
# Only after design-lock ratify + written pilot enable
export REVIEW_REUSE_DECISIONS_ENABLED=true
export REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER=true
export INTEGRATION_AUTH_MODE=required
export INTEGRATION_JWT_SECRET=...
export INTEGRATION_JWT_AUDIENCE=...
export INTEGRATION_JWT_ISSUER=...
export REVIEW_REUSE_STORE=filesystem
export REVIEW_REUSE_STORE_DIR=/secure/pilot/review_reuse_tasks
```

**Kill switch (decisions):**  
```bash
unset REVIEW_REUSE_DECISIONS_ENABLED
# or export REVIEW_REUSE_DECISIONS_ENABLED=false
# restart API
```

---

## 3. Operator checks (decision path)

1. Without Bearer JWT, with decisions+require_validated on:  
   `POST /api/v1/review-reuse/tasks/{id}/decision` → **403** `reviewer_not_validated`.
2. With valid JWT (`sub` + `tenant_id`): same POST → **200**, `human_decision.reviewer_id == sub`.
3. Decisions off: POST → **403** `decisions_disabled` regardless of JWT.
4. Never paste decision/audit bundles into training JSONL (R2 HOLD).

---

## 4. Filesystem store backup / cleanup

When `REVIEW_REUSE_STORE=filesystem`, task JSON lives under:

```text
{REVIEW_REUSE_STORE_DIR}/{tenant}/tasks/{task_id}.json
{REVIEW_REUSE_STORE_DIR}/{tenant}/idempotency.json
```

### Commands

```bash
# Backup (timestamped tarball under --out-dir)
make review-reuse-store-backup
# or:
python scripts/review_reuse_store_ops.py backup \
  --store-dir data/review_reuse_tasks \
  --out-dir data/review_reuse_backups

# List tenants (task count + age_days of newest task; add --json for machine output)
make review-reuse-store-list

# Dry-run cleanup: list tenants older than N days (by newest task mtime)
python scripts/review_reuse_store_ops.py cleanup \
  --store-dir data/review_reuse_tasks \
  --older-than-days 30 \
  --dry-run

# Apply cleanup (delete tenant directories past retention)
python scripts/review_reuse_store_ops.py cleanup \
  --store-dir data/review_reuse_tasks \
  --older-than-days 30
```

**Safety**

- Default cleanup is **dry-run** unless `--apply` is passed (script uses `--dry-run` default; make target documents both).
- Do not run cleanup against a live path without backup.
- Backups may contain customer-adjacent metadata (file names, scores); store off-git, same isolation rules as samples.

---

## 5. Residual human (not closed by this doc)

| ID | Item |
|---|---|
| R11 | Owner ratify L3 design-lock |
| R12 | Written enable of decisions for a named pilot window |
| Track C | Contacts, sample conversations, measured pilot |

---

## 6. Pilot env preflight (advisory)

Before any decision window, run the read-only preflight to print posture A/B/C and flag dangerous combos (decisions on without `REQUIRE_VALIDATED` or without `INTEGRATION_AUTH_MODE=required`). It **reads** env only and never sets `REVIEW_REUSE_DECISIONS_ENABLED`.

```bash
make review-reuse-preflight
# or: python scripts/review_reuse_pilot_preflight.py
# exit 0 = advisory OK; exit 2 = dangerous combo (fix before enabling decisions)
```

---

## 7. Quick link matrix

| Goal | Command / flag |
|---|---|
| Unit tests | `make test-review-reuse` |
| Offline archive demo | `make review-reuse-isolated-archive` |
| Pilot env preflight | `make review-reuse-preflight` |
| Store backup | `make review-reuse-store-backup` |
| Store cleanup dry-run | `make review-reuse-store-cleanup-dry` |
| Store tenant list | `make review-reuse-store-list` |
| Decisions | only with owner + JWT + require_validated |
