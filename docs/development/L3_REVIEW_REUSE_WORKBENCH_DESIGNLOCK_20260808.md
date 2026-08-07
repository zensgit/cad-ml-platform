# L3 Design-Lock — ReviewReuse Workbench (EvidencePack / Human Decision)

**Date**: 2026-08-08 · **Status**: PROPOSED (for-review; do NOT self-merge; owner ratifies)  
**Rigor**: L3 (`PRODUCT_STRATEGY.md` §7.1 — human decision ledger, audit export, customer-drawing task surface)  
**Grounded on**: `origin/main@7c9ae483` (post-#546 CAD Reuse Workbench 90-day plan)  
**Authority**:
- `docs/PRODUCT_STRATEGY.md` §3.3 (stable decision contract), §7.1 (L3), §8.2 (reuse / revise / new vertical slice), §8.3–§8.4 (pilot gates / calendar)
- `docs/development/CAD_REUSE_WORKBENCH_90_DAY_PLAN_20260807.md` (execution plan; Day 0 = 2026-07-12, Day 90 = 2026-10-10)

> **This document binds the strategy-minimum contract before (and with) the first runtime MVP.**  
> Runtime code in this tranche implements the bound surface with **decision submission default-off**.  
> Solo-maintainer L3 review protocol applies: an isolated critic supplies evidence; the human owner alone ratifies and pins a head. AI has **no release authority**.

---

## 0. Why this lock exists

Strategy §8.2 defines the product vertical slice:

```text
Upload / ingest → ReviewReuseTask → candidates → deterministic verification
  → EvidencePack → human reuse|revise|new → audit (no canonical write-back in MVP)
```

Without a design-lock, any ledger-writing decision sink or audit export is L3 risk under §7.1.
This lock freezes:

1. domain objects and field families (strategy-minimum first);
2. API surface under `/api/v1/review-reuse/*`;
3. tenant isolation and identity attribution rules;
4. default-off decision enablement;
5. quarantine of review evidence from training-readable paths;
6. explicit non-goals (no retrain unlock, no `eval_integrity_gate` replace, no model-release metrics).

---

## 1. Product boundary (non-negotiable)

| In scope (MVP) | Out of scope (this lock / 90-day runtime) |
|---|---|
| `ReviewReuseTask` lifecycle + step events | Generic CAD chatbot / WorkBuddy clone |
| EvidencePack export (JSON + Markdown) | Canonical PLM write-back / merge / delete |
| Human decision ledger (gated) | Retrain unlock or training-readable JSONL reuse |
| Tenant-isolated task store (MVP in-memory) | Multi-process durable store (later) |
| Isolated-archive task → EvidencePack demo | Track E model-run metrics / promotion gates |
| Structured rejection reasons | Hosted LLM egress for sample drawings |

**AI has no release authority.** Human reviewer or customer PLM remains the authority.

---

## 2. Domain objects

### 2.1 `ReviewReuseTask`

| Field | Type | Notes |
|---|---|---|
| `task_id` | UUID string | Primary key |
| `tenant_id` | string | Isolation boundary |
| `status` | enum | `pending` · `running` · `evidence_ready` · `decided` · `failed` · `canceled` |
| `created_at` / `updated_at` | unix float | Server clock |
| `source_file_name` | string | Display only |
| `source_content_sha256` | hex | Content integrity |
| `idempotency_key` | optional string | Create-time de-dupe per tenant |
| `trace_id` | UUID string | Cross-system correlation |
| `candidates` | list[`CandidateDecision`] | After recall/precision |
| `events` | list[`TaskEvent`] | Append-only for MVP |
| `evidence_pack` | object \| null | Built when ready |
| `human_decision` | `HumanDecision` \| null | Only after enabled submit |
| `calibration_version` | string | e.g. `workbench-mvp-0` |
| `error` | optional string | Failure detail |

### 2.2 `TaskEvent`

Event types (closed set for MVP):

`submitted` · `input_validated` · `recall_started` · `recall_completed` ·
`precision_started` · `precision_completed` · `evidence_pack_ready` ·
`decision_submitted` · `failed` · `canceled`

Each event: `{ event_type, ts, detail }`.

### 2.3 `CandidateDecision`

| Field | Notes |
|---|---|
| `candidate_id` | Stable within task |
| `candidate_source` | e.g. `archive`, `none` |
| `state` | `duplicate` · `similar` · `different` · `insufficient_evidence` |
| `scores` | At least `geometric`, `semantic` (nullable floats). **Optional extension:** `visual` |
| `verification` | `{ verdict, level, methods[] }` deterministic |
| `rejection_reasons` | Structured codes (below) |
| `provenance` | Model / input hash / query file |

**Structured rejection reasons (closed set for MVP):**

`missing_geom_json` · `version_gate_filtered` · `low_precision_score` ·
`vision_only_unverified` · `unsupported_file_type` ·
`external_service_unavailable` · `tool_unavailable`

### 2.4 `EvidencePack` (strategy §3.3 minimum)

| §3.3 field family | EvidencePack binding |
|---|---|
| candidate id + source | `candidates[].candidate_id`, `candidate_source` |
| normalized scores | `candidates[].scores.geometric/semantic` + `score_normalization` |
| deterministic verification | `candidates[].verification` |
| confidence + calibration | top-level `confidence`, `calibration.version` |
| evidence + rejection | `evidence[]`, `rejection_reasons[]`, `unsupported_states[]` |
| provenance | top-level `provenance` + per-candidate |
| human decision state | `human_decision.state`, `allowed_actions` |
| trace / idempotency | `task_id`, `trace_id`, `idempotency_key`, `source_job_id` |

**Implementation extension (optional, non-blocking):** `scores.visual`.  
Must not block design-lock ratification or EvidencePack export.

Export formats:

- **JSON** — machine-readable (default).
- **Markdown** — reviewer report (`format=markdown` query).

### 2.5 `HumanDecision`

**Strategy-center states:** `reuse` · `revise` · `new`  

**Implementation extensions** (must not blur with CandidateDecision states):

- `reject_candidate`
- `need_more_info`

Fields: `state`, `reviewer_id`, `reason_codes[]`, `reason_text`, `candidate_id?`, `ts`, `idempotency_key?`.

---

## 3. API surface

Full paths (repo mounts `api_router` at `/api`, `v1_router` at `/v1`):

| Method | Path | Notes |
|---|---|---|
| POST | `/api/v1/review-reuse/tasks` | multipart file upload; optional form `idempotency_key` |
| GET | `/api/v1/review-reuse/tasks` | list for tenant |
| GET | `/api/v1/review-reuse/tasks/{task_id}` | get |
| POST | `/api/v1/review-reuse/tasks/{task_id}/cancel` | cancel |
| GET | `/api/v1/review-reuse/tasks/{task_id}/events` | event log |
| GET | `/api/v1/review-reuse/tasks/{task_id}/evidence-pack` | `?format=json\|markdown` |
| POST | `/api/v1/review-reuse/tasks/{task_id}/decision` | **gated** |

Auth: existing `get_api_key` dependency.  
Tenant: `request.state.tenant_id` if set, else stable hash of API key (`ak-<16hex>`).  
Reviewer: JWT subject / user_id if present; else `ak-user-<12hex>` (not trusted pilot identity — pilot requires validated identity).

---

## 4. Enablement — human decision sink

| Control | Default | Behavior |
|---|---|---|
| `REVIEW_REUSE_DECISIONS_ENABLED` | **off** (unset / false) | POST decision → **403** `decisions_disabled` |
| `REVIEW_REUSE_DECISIONS_ENABLED=true\|1\|yes\|on` | owner opt-in | Accept strategy-center + extension states |

**Owner enable step (Day 61–90 pilot):** set the env flag in the pilot deployment only after L3 acceptance tests pass. This is **not** a retrain-gate change and does **not** touch `eval_integrity_gate`.

Idempotency: same decision `idempotency_key` on an already-decided task returns the existing decision; different key → **409** `already_decided`.

---

## 5. Tenant isolation and audit boundaries

1. Task lookup is always `(tenant_id, task_id)`. Cross-tenant access returns **404** (not 403) to avoid existence leaks.
2. MVP store is **process-local in-memory** — not multi-process durable. Pilot Day 61–90 may replace store without changing the API contract.
3. **Quarantine:** decision / correction evidence MUST NOT enter training-readable manifests. Do **not** reuse `src/api/v1/feedback.py` JSONL training path as the ledger store.
4. **No canonical write-back** in MVP (no PLM mutate).
5. Audit export = EvidencePack JSON/Markdown + task events. Deletion/retention: see isolated-sample runbook.

---

## 6. Pipeline semantics (MVP)

1. Create task → `submitted` → `input_validated` → `running`.
2. `recall_*` → candidates (seeded fixture **or** offline single `insufficient_evidence` row when no tool).
3. `precision_*` → verification fields populated.
4. Build EvidencePack → `evidence_ready` + `evidence_pack_ready`.
5. Optional human decision (if enabled) → `decided` + `decision_submitted`; pack refreshed with decision.

**Honest offline path:** without a live dedup tool or seed, export still works with `tool_unavailable` / `insufficient_evidence`. Never invent a high-confidence duplicate.

---

## 7. Safety invariants (must remain true)

| ID | Invariant |
|---|---|
| R1 | Decision sink default-off |
| R2 | No training-path reuse for decisions |
| R3 | Tenant isolation on all task routes |
| R4 | EvidencePack always includes §3.3 minimum fields |
| R5 | No retrain unlock; no `eval_integrity_gate` mutation |
| R6 | No model-release metrics from this surface (Track E boundary) |
| R7 | No silent revival of `cost_cap` / day-90 residual board (#545) |
| R8 | AI has no release authority — human/PLM decides |

Baseline audits (preserve, do not rebuild):

- #537 assistant SEAL
- #538 production identity fail-closed
- #542 Track E Slice E1 (dry-run / manifest; not model-run metrics)

---

## 8. Isolated sample / archive rules (summary)

Full checklist: `docs/development/ISOLATED_SAMPLE_ARCHIVE_RUNBOOK_20260808.md`.

Minimum:

- no shared multi-tenant processing of customer samples;
- no hosted LLM provider egress for sample processing;
- documented storage, retention, deletion;
- before pilot gates pass: no production/shared-tenant customer drawing processing.

---

## 9. Acceptance for runtime MVP (Day 31–60 interpretation)

- [x] Design-lock exists and binds §3.3 fields.
- [x] API routes mounted under `/api/v1/review-reuse/*`.
- [x] Task events cover the closed set.
- [x] EvidencePack JSON + Markdown export.
- [x] Decision default-off with explicit env enable.
- [x] Tenant isolation tests.
- [x] Isolated-archive / synthetic end-to-end task → EvidencePack (service path with seed candidates).
- [ ] Owner ratification of this design-lock (human).
- [ ] Customer Track C residual (contacts / samples) — **not** engineering-blocking.

---

## 10. Residuals (explicit)

| Item | Owner | Window |
|---|---|---|
| Durable multi-process task store | Eng | Day 61–90 if pilot needs HA |
| Live dedup adapter (not offline seed) | Eng | Day 31–60 follow-up |
| Validated JWT reviewer identity for pilot | Eng + Ops | Day 61–90 |
| 10 manufacturer contacts + 2 sample conversations | Customer / Owner | residual into Day 31–60 |
| Measured pilot metrics package | Owner | Day 61–90 |
| E1 post-merge audit note at exact head | Eng | residual, non-blocking for design-lock |

---

## 11. Ratification

Owner must confirm in writing:

1. Workbench = strategy §3.3 + §8.2 (not generic assistant expansion).
2. Day-90 clock remains 2026-10-10 unless strategy is amended.
3. This design-lock is the product design document for Track R.
4. Decision enablement is env-gated default-off.
5. #507 portfolio questions do not block Track R while workbench is the single product track through Day 90.

**Status after merge of runtime PR:** still **owner-ratify** for pilot enablement of `REVIEW_REUSE_DECISIONS_ENABLED`.
