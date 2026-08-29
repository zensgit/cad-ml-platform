# L3 Design-Lock — ReviewReuse Workbench (EvidencePack / Human Decision)

**Date**: 2026-08-08 (security amendment 2026-08-29) · **Status**: PROPOSED (for-review; do NOT self-merge; owner ratifies)
**Rigor**: L3 (`PRODUCT_STRATEGY.md` §7.1 — human decision ledger, audit export, customer-drawing task surface)  
**Grounded on**: `origin/main@22e3c77c` (post-#565; amendment prepared in #583)
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
| Tenant-isolated task store (memory default; filesystem opt-in) | Multi-process / HA store (later) |
| Isolated-archive task → EvidencePack demo | Track E model-run metrics / promotion gates |
| Structured rejection reasons | Hosted LLM egress for sample drawings |

**AI has no release authority.** Human reviewer or customer PLM remains the authority.

---

## 2. Domain objects

### 2.1 `ReviewReuseTask`

| Field | Type | Notes |
|---|---|---|
| `task_id` | UUID string | Primary key |
| `tenant_id` | string | Literal isolation boundary; storage mapping MUST be collision-safe and non-escaping |
| `status` | enum | `pending` · `running` · `evidence_ready` · `decided` · `failed` · `canceled` |
| `created_at` / `updated_at` | unix float | Server clock |
| `source_file_name` | string | Display only |
| `source_content_sha256` | hex | Content integrity |
| `idempotency_key` | optional string | Create-time de-dupe per tenant |
| `idempotency_digest` | optional hex | Canonical create payload digest bound to the key |
| `revision` | integer | Starts at 1; increments on every persisted mutation |
| `trace_id` | UUID string | Cross-system correlation |
| `candidates` | list[`CandidateDecision`] | After recall/precision |
| `events` | list[`TaskEvent`] | Append-only for MVP |
| `evidence_pack` | object \| null | Built when ready |
| `human_decision` | `HumanDecision` \| null | Only after enabled submit |
| `calibration_version` | string | e.g. `workbench-mvp-0` |
| `error_code` / `error` | optional string | Structured failure code + safe detail |

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

Fields: `state`, `reviewer_id`, `reason_codes[]`, `reason_text`, `candidate_id?`, `ts`, `idempotency_key?`, `idempotency_digest?`.

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
| GET | `/api/v1/review-reuse/metrics` | Review-workflow metrics only; not Track E model-release metrics |
| GET | `/api/v1/review-reuse/tasks/{task_id}/audit-export` | Full quarantined task/event/EvidencePack snapshot; not training data |
| POST | `/api/v1/review-reuse/tasks/{task_id}/decision` | **gated** |

Auth: existing `get_api_key` dependency.  
Tenant: `request.state.tenant_id` if set, else stable hash of API key (`ak-<16hex>`).  
Reviewer: JWT subject / user_id if present; else `ak-user-<12hex>` (not trusted pilot identity — pilot requires validated identity).

Canonical errors:

| Code | HTTP | Meaning |
|---|---:|---|
| `tenant_invalid` | 400 | Tenant identity violates §5 |
| `empty_input` / `invalid_decision` | 422 | Input or decision contract violation |
| `input_too_large` | 413 | Upload exceeds 50 MiB |
| `unsupported_file_type` | 415 | MVP input is not `.dxf` |
| `not_found` | 404 | No task visible to this tenant |
| `decisions_disabled` / `reviewer_not_validated` | 403 | Owner or identity gate closed |
| `not_ready` / `invalid_state_transition` | 409 | Requested operation is invalid for current state |
| `already_decided` / `idempotency_key_conflict` / `revision_conflict` | 409 | Ledger conflict |
| `store_index_corrupt` / `store_record_corrupt` | 503 | Persistent ledger integrity unavailable |

Unlisted domain errors MUST NOT silently collapse to catch-all HTTP 400.

---

## 4. Enablement — human decision sink

| Control | Default | Behavior |
|---|---|---|
| `REVIEW_REUSE_DECISIONS_ENABLED` | **off** (unset / false) | POST decision → **403** `decisions_disabled` |
| `REVIEW_REUSE_DECISIONS_ENABLED=true\|1\|yes\|on` | owner opt-in | Accept strategy-center + extension states |
| `REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER` | **off** | Pilot preflight requires owner-approved validated identity posture |
| `REVIEW_REUSE_LIVE_DEDUP` | **off** | Offline `insufficient_evidence`; live evidence is not enabled implicitly |
| `REVIEW_REUSE_STORE` | `memory` | `filesystem` is not pilot-eligible until §5 items 1-5 pass |
| `REVIEW_REUSE_STORE_DIR` | `data/review_reuse_tasks` | Used only by the filesystem backend; path remains deployment-controlled |
| `REVIEW_REUSE_MAX_UPLOAD_BYTES` | `52428800` (50 MiB) | Streaming upload cap; no unbounded `read()` |

**Owner enable step (Day 61–90 pilot):** set the env flag in the pilot deployment only after L3 acceptance tests pass. This is **not** a retrain-gate change and does **not** touch `eval_integrity_gate`.

### 4.1 Canonical idempotency

The key is a retry identity, not the payload identity. It MUST be bound to a persisted
`sha256(canonical_json(payload))` digest:

- Create payload: `{tenant_id, source_content_sha256, source_file_name}`.
- Decision payload: `{tenant_id, task_id, state, candidate_id, sorted(reason_codes), reason_text, reviewer_id}`.

Same key + same digest returns the stored result. Same key + different digest returns
**409** `idempotency_key_conflict`. A different or absent key on an already-decided task
returns **409** `already_decided`. Keys are tenant/surface scoped, printable, and at most
128 characters.

### 4.2 Atomic ledger mutation

All persisted mutations use `revision` compare-and-set. Exactly one concurrent decision
may commit; the loser receives **409** `revision_conflict` or `already_decided`. A commit
MUST NOT reduce the event list. Process-local locks do not satisfy multi-process atomicity;
the filesystem pilot is single-writer until an independently reviewed locking design exists.

---

## 5. Tenant isolation and audit boundaries

1. Task lookup is always `(tenant_id, task_id)`. Cross-tenant access returns **404** (not 403) to avoid existence leaks.
2. `tenant_id` MUST match `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`, MUST NOT be `.` or `..`, and is rejected as **400** `tenant_invalid` before any store call otherwise. Lossy replacement or truncation is forbidden. A path backend uses the full SHA-256 of the literal UTF-8 identity as the directory segment, records the literal identity in the task, verifies the resolved path remains under the configured root, and rechecks stored `tenant_id` + `task_id` on every load.
3. Existing lossy filesystem directories are legacy. Migration enumerates embedded tenant identities and aborts on collisions or mismatches; ambiguous directories MUST NOT be merged. Until that migration and its tests pass, filesystem storage is not pilot-eligible.
4. A malformed idempotency index fails closed with **503** `store_index_corrupt`; it is never treated as empty or overwritten. A malformed owning-tenant record returns **503** `store_record_corrupt`; a cross-tenant caller still receives 404. List/metrics report unreadable records instead of silently dropping them.
5. Writes use unique temporary files and durable atomic replacement. Task/index ordering must leave a recoverable task record after interruption. Retention refuses records that fail stored identity checks.
6. **Quarantine:** decision / correction evidence MUST NOT enter training-readable manifests. Do **not** reuse `src/api/v1/feedback.py` JSONL training path as the ledger store.
7. **No canonical write-back** in MVP (no PLM mutate).
8. Audit export = full task snapshot + EvidencePack JSON/Markdown + task events. Deletion/retention: see isolated-sample runbook.

---

## 6. Pipeline semantics (MVP)

1. Stream and validate a non-empty `.dxf` input up to 52,428,800 bytes. Only then create task → `submitted` → `input_validated` → `running`.
2. `recall_*` → candidates (seeded fixture **or** offline single `insufficient_evidence` row when no tool).
3. `precision_*` → verification fields populated.
4. Build EvidencePack → `evidence_ready` + `evidence_pack_ready`.
5. Optional human decision (if enabled) → `decided` + `decision_submitted`; pack refreshed with decision.

**Honest offline path:** without a live dedup tool or seed, export still works with `tool_unavailable` / `insufficient_evidence`. Never invent a high-confidence duplicate.

Input failures use `empty_input`, `input_too_large`, or `unsupported_file_type`, are
rejected before task persistence, and MUST NOT emit `submitted` or `input_validated`.
Downstream recall/precision/evidence failures after persistence set terminal `failed`,
populate `error_code` + safe `error`, and append a `failed` event.

Legal persisted transitions are `pending → running → evidence_ready → decided`,
`{pending,running} → failed`, and `{pending,running,evidence_ready} → canceled`.
Only `evidence_ready` may be decided. `decided`, `failed`, and `canceled` are terminal.

### 6.1 Human decision matrix

| State | Candidate | Rationale |
|---|---|---|
| `reuse` | required and present in task | non-empty reason code(s) or text |
| `revise` | required and present in task | non-empty reason code(s) or text |
| `new` | absent | non-empty reason code(s) or text |
| `reject_candidate` | required and present in task | non-empty reason code(s) or text |
| `need_more_info` | optional; if present, present in task | non-empty reason code(s) or text |

The decision reason vocabulary is closed and distinct from candidate rejection reasons:
`geometry_match`, `visual_similarity_only`, `needs_modification`, `new_part_required`,
`insufficient_evidence`, `incorrect_candidate`, and `other`. `other` requires non-empty text.
Any candidate-bound decision against `insufficient_evidence` is an explicit override and
must remain visible in the refreshed EvidencePack.

### 6.2 Honest score and evidence semantics

1. A score dimension is populated only when independently produced. One visual similarity MUST NOT be copied into `geometric` or `semantic`; missing values remain null.
2. Candidate evidence records score producer/version/raw field. `score_normalization` names a transform actually applied, otherwise `none`.
3. With geometry disabled or unavailable, emit `vision_only_unverified`; do not claim a geometric precision level or `duplicate`. Without deterministic verification the maximum state is `similar`.
4. Confidence is not `max()` across score aliases. It records score, band, source dimension, method, and `verified`; unverified evidence cannot be `high`, and all-insufficient evidence has null score + low band.
5. Calibration records `{version,status}` where status is `uncalibrated` or `calibrated`. No fitted-sounding version is emitted without a real artifact.
6. Provenance contains executed-path values: tool/version, archive id, index digest, ruleset version, query SHA, and path (`seed_fixture`, `live_dedup`, or `offline_stub`). Seed fixtures are marked `synthetic: true`.
7. Unknown tool verdicts map to `insufficient_evidence`, never silently to `similar`.
8. JSON and Markdown exports both disclose confidence verification, unsupported states, rejection reasons, provenance path, and synthetic status.

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
| R9 | Tenant storage is collision-safe, root-contained, and identity-checked on load |
| R10 | Task/event/decision mutation is revision-CAS atomic |
| R11 | Scores, confidence, calibration, and provenance disclose only executed evidence |
| R12 | Corrupt store/index artifacts fail closed and remain observable |
| R13 | Reviewer UI remains within the L2/L3 boundary in §12 |

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
- [ ] Task transition/failure events satisfy §6 and atomic ledger semantics.
- [ ] EvidencePack JSON + Markdown satisfy §6.2 honest-evidence parity.
- [x] Decision default-off with explicit env enable.
- [ ] Adversarial tenant collision/root-escape/load-identity tests.
- [ ] First real isolated archive/index/query/replay without seeded candidates.
- [ ] Owner ratification of this design-lock (human).
- [ ] Customer Track C residual (contacts / samples) — **not** engineering-blocking.

---

## 10. Residuals (explicit)

| Item | Owner | Window |
|---|---|---|
| ER1 collision-safe filesystem store + legacy migration | Eng | blocked on this lock ratification |
| ER2 input/idempotency/CAS decision ledger | Eng | after ER1 under the same L3 safety tranche |
| ER3 real archive + honest EvidencePack replay | Eng + Owner | after ER1/ER2 and approved isolated data |
| ER4 minimal reviewer workbench | Eng | after ER1-ER3; §12 boundary applies |
| Durable multi-process / HA store | Eng | optional future; not ER1 closure |
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
6. Tenant path, migration, record-integrity, and fail-closed store rules in §5 are approved.
7. Canonical create/decision idempotency and revision-CAS semantics in §4 are approved.
8. Honest score/provenance/calibration semantics supersede current fixtures and goldens.
9. The reviewer UI L2/L3 boundary in §12 is approved.
10. Ratification pins an exact document head and does not enable `REVIEW_REUSE_DECISIONS_ENABLED`.

**Status after merge of runtime PR:** still **owner-ratify** for pilot enablement of `REVIEW_REUSE_DECISIONS_ENABLED`.

---

## 12. Reviewer workbench UI boundary

**L2 after ER1-ER3:** read-only task queue/detail, event timeline, candidate/EvidencePack
comparison, quarantined export download, and a pass-through decision form that is visibly
disabled when the server gate is off. The UI renders unsupported, unverified, synthetic,
and unavailable states; it does not hide them or recompute scores.

**Re-escalates to L3:** any new auth/session/identity path, client-side ledger or durable
state, decision/default-off semantic change, score/confidence recomputation or ranking,
LLM/assistant surface, canonical write-back, or new export destination.

The UI MUST NOT preselect a human decision, supply a default rationale, place API keys in
browser persistent storage, load external CDN assets during isolated-sample work, or imply
AI/release authority. A future read-only capabilities endpoint and list pagination may be
added under this lock, but implementation stays deferred until ER1-ER3 close.
