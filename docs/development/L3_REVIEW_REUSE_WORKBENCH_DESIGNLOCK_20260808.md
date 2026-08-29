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
| `calibration_version` | string \| null | Real fitted artifact/version only; otherwise null |
| `calibration_status` | enum | `uncalibrated` · `calibrated` |
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
| reviewed version | `task_revision`, `evidence_pack_sha256` |

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

Fields: `state`, `reviewer_id`, `reviewer_kind`, `reason_codes[]`, `reason_text`, `candidate_id?`, `ts`,
`idempotency_key?`, `idempotency_digest?`, `reviewed_revision`, and
`evidence_pack_sha256`. The last two bind the decision to the evidence actually reviewed.

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
| GET | `/api/v1/review-reuse/capabilities` | Read-only ER4 gate/identity posture; no key material or tenant inventory |

### 3.1 Decision request wire contract

The decision POST uses a JSON body, not `If-Match`:

```json
{
  "state": "reuse",
  "candidate_id": "candidate-1",
  "reason_codes": ["geometry_match"],
  "reason_text": "Reviewed geometry and provenance.",
  "idempotency_key": "client-retry-key",
  "expected_revision": 4,
  "evidence_pack_sha256": "<64 lowercase hex characters>"
}
```

`expected_revision` is a required integer greater than or equal to 1 and
`evidence_pack_sha256` is a required 64-character lowercase hexadecimal string.
Missing or malformed fields fail request validation with **422** `invalid_request` (or
`invalid_decision` for a semantically invalid decision). A well-formed request whose
revision or digest no longer matches the stored task fails with **409**
`revision_conflict`. The server derives `tenant_id`, `task_id`, and `reviewer_id`; clients
cannot submit or override them.

Auth: existing `get_api_key` dependency.  
Tenant: validated `request.state.tenant_id` if set, else stable hash of API key
(`ak-<16hex>`).
Reviewer: a validated identity becomes
`principal-v1-<64 lowercase hex SHA-256 of
canonical_json_v1({identity_provider, subject})>` with
`reviewer_kind="validated_principal"`. `identity_provider` is the exact configured
`INTEGRATION_JWT_ISSUER` value after the JWT `iss` claim has matched it, and `subject` is the
exact decoded JWT `sub` string. Empty values or leading/trailing whitespace in either value
are invalid rather than normalized. Both values must come from verified middleware; if
either is absent the identity is not validated. API-key fallback is
`ak-user-<12hex>` with `reviewer_kind="api_key_fallback"`; it is read/create-only and can
never submit a decision. Clients cannot supply either field.

Canonical errors:

| Code | HTTP | Meaning |
|---|---:|---|
| platform authentication response | 401 | Upstream API-key/JWT authentication failed; existing `{detail: string}` platform contract, before ReviewReuse domain handling |
| `tenant_invalid` | 400 | Tenant identity violates §5 |
| `empty_input` / `invalid_decision` / `invalid_request` | 422 | Input, decision, or framework request validation failure |
| `input_too_large` | 413 | Upload exceeds 50 MiB |
| `unsupported_file_type` | 415 | MVP input is not `.dxf` |
| `not_found` | 404 | No task visible to this tenant |
| `decisions_disabled` / `tenant_not_validated` / `reviewer_not_validated` | 403 | Owner or identity gate closed |
| `not_ready` / `invalid_state_transition` | 409 | Requested operation is invalid for current state |
| `already_decided` / `idempotency_key_conflict` / `revision_conflict` | 409 | Ledger conflict |
| `store_index_corrupt` / `store_record_corrupt` / `store_writer_conflict` | 503 | Persistent ledger integrity or single-writer lease unavailable |
| `internal_error` | 500 | Sanitized unexpected failure; details remain operator-only |

After successful platform authentication, framework validation MUST use the same
`{detail: {code, message}}` ReviewReuse envelope. The upstream 401 row is intentionally not
rewritten by this lock; it is documented in the route OpenAPI as the existing platform-auth
boundary. Unlisted domain errors MUST NOT silently collapse to catch-all HTTP 400; they fail
closed as a sanitized 500 `internal_error`. Runtime aliases such as `tenant_required`,
`reviewer_required`, and `canceled` are replaced by `tenant_invalid`, `invalid_decision`,
and `invalid_state_transition` rather than extending the public vocabulary.

---

## 4. Enablement — human decision sink

| Control | Default | Behavior |
|---|---|---|
| `REVIEW_REUSE_DECISIONS_ENABLED` | **off** (unset / false) | POST decision → **403** `decisions_disabled` |
| `REVIEW_REUSE_DECISIONS_ENABLED=true\|1\|yes\|on` | owner opt-in | Accept strategy-center + extension states |
| `REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER` | legacy compatibility only | Cannot weaken the mandatory validated tenant + reviewer decision rule |
| `REVIEW_REUSE_LIVE_DEDUP` | **off** | Offline `insufficient_evidence`; live evidence is not enabled implicitly |
| `REVIEW_REUSE_STORE` | `memory` | `filesystem` is not pilot-eligible until §5 items 1-7 pass |
| `REVIEW_REUSE_STORE_DIR` | `data/review_reuse_tasks` | Used only by the filesystem backend; path remains deployment-controlled |
| `REVIEW_REUSE_MAX_UPLOAD_BYTES` | `52428800` (50 MiB) | Streaming upload cap; no unbounded `read()` |

**Owner enable step (Day 61–90 pilot):** set the decision env flag in the pilot deployment
only after L3 acceptance tests pass. Decision submission additionally requires a validated
tenant claim and validated reviewer principal on every request; API-key-derived fallback
identities are read/create-only and MUST receive 403 `reviewer_not_validated`. Setting the
legacy reviewer env false never bypasses this rule, and a decision-enabled deployment that
cannot establish both principals fails preflight/startup. This is **not** a retrain-gate
change and does **not** touch `eval_integrity_gate`.

### 4.1 Canonical idempotency

The key is a retry identity, not the payload identity. It MUST be bound to a persisted
`sha256(canonical_json_v1(payload))` digest. `canonical_json_v1` is RFC 8785 JSON
Canonicalization Scheme (JCS): UTF-8, deterministic property sorting and escaping, no
insignificant whitespace, arrays in their specified order, and ECMAScript-compatible
binary64 number serialization. `NaN`, positive/negative infinity, duplicate object keys,
and non-I-JSON values are rejected before persistence; negative zero serializes as `0`.
Before encoding, trim surrounding whitespace from enumerated/string identifiers and
rationale text, preserve internal rationale whitespace, and sort/deduplicate
`reason_codes` by code-point order.

Every field named in a digest payload is present. Nullable fields such as
`candidate_id` are encoded as JSON `null`, not omitted; collection fields such as
`reason_codes` are encoded as arrays, including `[]`. EvidencePack hashing is the one
explicit exclusion rule: remove the `evidence_pack_sha256` member, then run
`canonical_json_v1` over the remaining object. Setting that member to `null` is not
equivalent.

An EvidencePack is materialized and persisted exactly once for each task revision; GET,
Markdown export, audit export, and digest verification read that persisted object rather
than rebuilding it from mutable models, archive contents, clocks, or random values. Thus a
given `(task_id, task_revision)` has exactly one byte-reproducible canonical pack. Any
evidence refresh is a CAS mutation to a new revision and produces a new digest.

- Create payload: `{tenant_id, source_content_sha256}`. `source_file_name` is display-only;
  a same-content retry under a different name returns the original stored result/name.
- Decision payload: `{tenant_id, task_id, state, candidate_id, sorted(reason_codes),
  reason_text, reviewer_id, reviewer_kind, expected_revision, evidence_pack_sha256}`.

Same key + same digest returns the stored result. Same key + different digest returns
**409** `idempotency_key_conflict`. A different or absent key on an already-decided task
returns **409** `already_decided`. Keys are tenant/surface scoped, printable, and at most
128 characters.

Decision idempotency is actor-bound: `reviewer_id` and `reviewer_kind` are the canonical
authenticated principal recorded in the ledger. The `principal-v1-` and `ak-user-`
namespaces cannot alias, and two identity providers with the same subject hash differently.
A retry under a different principal, including an API-key fallback to validated identity
change, is intentionally a different payload and returns 409. The client
must obtain a new key after an identity change; it never rewrites original attribution.
Before commit, `expected_revision` and `evidence_pack_sha256` MUST match the current task and
canonical EvidencePack bytes; stale evidence returns 409 `revision_conflict`.
The pack exposes `task_revision` and `evidence_pack_sha256`; the digest is
`sha256(canonical_json_v1(pack after deleting the evidence_pack_sha256 member))`. After
commit, the refreshed
pack records the decision's `reviewed_revision` and reviewed digest rather than pretending
the reviewer saw the post-decision pack.

### 4.2 Atomic ledger mutation

All persisted mutations use a store-level compare-and-set primitive
`put(task, expected_revision=...)`; a process `RLock` or service-level `get()` + `put()`
sequence is not CAS. The first persisted task snapshot is revision 1 and every successful
mutation increments by exactly one. Exactly one concurrent decision may commit; the loser
receives **409** `revision_conflict` or `already_decided`. A commit MUST NOT reduce or replace
the prior event prefix. Cancel and decision contend on the same expected revision; exactly
one wins and the loser receives 409 without appending an event.

Idempotent create uses a store-level `create_if_absent(task, key, digest)` critical section,
not a service `get_by_idempotency()` + `put()` sequence. The task record containing key and
digest is the source of truth and is atomically replaced before the index. On restart, a
missing index entry is resolved by scanning identity-valid records under the writer lease:
zero matching records is the normal first use and may create one task; exactly one record
with the same key and digest is replayed and its index entry rebuilt; duplicate, mismatched,
or malformed ownership fails closed instead of creating a second task. Same key/digest has
one durable task id under concurrent calls and crash recovery.

The filesystem pilot is single-writer. A write-capable API process MUST acquire and hold a
non-blocking, process-lifetime exclusive writer lease under the store root before accepting
mutations; failure returns/starts unhealthy with **503** `store_writer_conflict`. Read-only
ops/export processes use an explicit read-only store mode. Multi-writer or inherited/forked
worker behavior remains out of scope until an independently reviewed locking design exists.

---

## 5. Tenant isolation and audit boundaries

1. Task lookup is always `(tenant_id, task_id)`. `task_id` is a server-generated canonical
   lowercase hyphenated UUID. Every external path value must parse and round-trip to that
   canonical form before a store call; malformed/non-canonical values return **404**
   `not_found`. Task records exist only as `tasks/<task_id>.json`, while tenant metadata and
   the idempotency index remain outside `tasks/`, so task names cannot collide with store
   metadata. Cross-tenant access returns **404** (not 403) to avoid existence leaks. Decision
   writes additionally require validated tenant and reviewer principals; fallback API-key
   identities cannot write the ledger.
2. `tenant_id` MUST match `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`, MUST NOT be `.` or `..`, and is rejected as **400** `tenant_invalid` before any store call otherwise. Lossy replacement or truncation is forbidden. `ak-` is a reserved API-key-fallback namespace: a trusted tenant claim beginning with that prefix is rejected. A path backend uses `tenant-v1-<64 lowercase hex SHA-256 of literal UTF-8 identity>` as the directory segment (the prefix separates the new layout from a 64-hex legacy literal), verifies the resolved path remains under the configured root, and rechecks sidecar + stored `tenant_id` + `task_id` on every load. `get`, list, metrics, and idempotency reads never create directories or sidecars.
3. Each new-layout tenant directory has exactly one canonical `tenant.json` sidecar:
   `{schema_version: "review-reuse-tenant-v1", tenant_id: <literal>, identity_source:
   "validated_claim"|"api_key_fallback", tenant_digest_sha256: <64 lowercase hex SHA-256 of literal UTF-8
   tenant_id>}`. All four fields are required; unknown fields, a directory/digest mismatch,
   or an identity-source mismatch fail closed as **503** `store_record_corrupt`. The sidecar
   contains no API key, token, reviewer identity, or other secret.
4. The per-tenant create-idempotency index has exactly
   `{schema_version: "review-reuse-idempotency-v1", tenant_id: <literal>,
   tenant_digest_sha256: <same sidecar digest>, entries: {<key>: {task_id: <UUID>,
   payload_digest: <64 lowercase hex>}}}`. All fields and entry ownership are validated on
   read; an entry whose task record does not carry the same tenant, key, digest, and task id
   is corrupt. Unknown fields or any schema/identity/ownership mismatch return **503**
   `store_index_corrupt`; they are never repaired by an ordinary request.
5. Existing lossy filesystem directories are legacy. Migration is backup-first and dry-run by default; it enumerates embedded tenant identities and aborts before writes on collisions or mismatches. Ambiguous directories MUST NOT be merged. A valid legacy record is upgraded to `revision=1`, `error_code=null`, and reconstructed create/decision digests using `canonical_json_v1`; missing identity/hash/decision fields or an unprovable index mapping abort migration. Legacy decided records remain immutable even when their historical reason code is outside the new submission vocabulary. Apply uses a staging directory plus atomic rename and is restart-verifiable. Until migration and tests pass, filesystem storage is not pilot-eligible.
6. A malformed idempotency index fails closed with **503** `store_index_corrupt`; it is never treated as empty or overwritten. A malformed owning-tenant record returns **503** `store_record_corrupt`; a cross-tenant caller still receives 404. List and metrics return 503 rather than partial results when any owning-tenant record is unreadable; sanitized corruption counts/paths go to operator logs/metrics, not the tenant response.
7. Writes require the §4.2 writer lease and use unique temporary files plus durable atomic replacement. Task record is replaced before its index; an index failure marks the store unhealthy and blocks retries until repair/rebuild, preventing a duplicate task. Retention, backup, list, preflight, and migration resolve literal tenant identity through the sidecar and refuse records that fail stored identity checks.
8. **Quarantine:** decision / correction evidence MUST NOT enter training-readable manifests. Do **not** reuse `src/api/v1/feedback.py` JSONL training path as the ledger store.
9. **No canonical write-back** in MVP (no PLM mutate).
10. Audit export is available for every persisted status. It contains the full task snapshot,
   task events, and `evidence_pack` JSON/Markdown when present; pending/running/failed tasks
   export `evidence_pack=null`, empty Markdown, plus structured `error_code`/safe `error`.
   Deletion/retention: see isolated-sample runbook.

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
Downstream recall/precision/evidence failures while `pending` or `running` set terminal
`failed`, populate `error_code` + safe `error`, and append a `failed` event. Decision
validation and refreshed-pack construction occur before the decision CAS; a pre-commit
failure leaves the prior `evidence_ready` snapshot/event prefix/revision unchanged and
returns sanitized `internal_error`, so the reviewer may safely retry.

Legal persisted transitions are `pending → running → evidence_ready → decided`,
`{pending,running} → failed`, and `{pending,running,evidence_ready} → canceled`.
Only `evidence_ready` may be decided.
`decided`, `failed`, and `canceled` are terminal. Cancel is retry-idempotent only for an
already-canceled task: return the unchanged task with 200, without a new event/revision.
Cancel against `decided` or `failed` returns 409 `invalid_state_transition`.

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

`need_more_info` is a terminal triage outcome for the MVP: it persists as `decided` and a
later follow-up creates a new linked task outside this lock; it is not a hidden reopen path.

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
- [ ] Canonical idempotency, create recovery, store-level CAS, writer-lease, corruption and migration tests.
- [ ] Validated tenant/reviewer and reviewed EvidencePack revision/digest decision binding.
- [ ] Canonical HTTP responses plus regenerated OpenAPI snapshot/contract test.
- [ ] First real isolated archive/index/query/replay without seeded candidates.
- [ ] Owner ratification of this design-lock (human).
- [ ] Customer Track C residual (contacts / samples) — **not** engineering-blocking.

### 9.1 Required fail-first tranche

Before any ER1/ER2 runtime file changes, commit these named tests and retain a baseline log
showing they fail against the ratified design-lock head:

- `tests/unit/test_review_reuse_er1_store_integrity.py`:
  `test_tenant_segments_do_not_collide`, `test_dotdot_cannot_escape_store_root`,
  `test_reads_do_not_create_tenant_paths`, `test_loaded_identity_mismatch_fails_closed`,
  `test_corrupt_index_and_record_fail_closed`, `test_legacy_migration_aborts_on_collision`,
  `test_second_writer_is_rejected`, `test_create_if_absent_recovers_one_task`.
- `tests/unit/test_review_reuse_er2_ledger.py`:
  `test_empty_oversized_and_unsupported_input_do_not_persist`,
  `test_decision_requires_validated_tenant_and_reviewer`,
  `test_decision_requires_current_revision_and_evidence_digest`,
  `test_idempotency_key_conflicts_on_payload_change`,
  `test_decision_matrix_and_reason_vocabulary`, `test_cancel_retry_is_idempotent`,
  `test_concurrent_decision_and_cancel_commit_once`.
- `tests/unit/test_review_reuse_api_integrity.py`:
  `test_canonical_error_status_and_envelope`, `test_decision_revision_fields_are_required`,
  `test_store_corruption_maps_to_503`.

The implementation PR must attach the fail-first command/output before showing the same
tests green. Renaming, weakening, deleting, or marking these tests xfail/skip requires owner
review; adding narrower tests is allowed.

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
10. Canonical JSON, terminal/cancel semantics, legacy defaults, create recovery, writer lease, and OpenAPI error contract are approved.
11. Decision writes require validated tenant/reviewer plus the reviewed revision/EvidencePack digest.
12. Ratification pins an exact document head and does not enable `REVIEW_REUSE_DECISIONS_ENABLED`.

**Status after merge of runtime PR:** still **owner-ratify** for pilot enablement of `REVIEW_REUSE_DECISIONS_ENABLED`.

---

## 12. Reviewer workbench UI boundary

**L2 after ER1-ER3:** read-only task queue/detail, event timeline, candidate/EvidencePack
comparison, quarantined export download, and a pass-through decision form that is visibly
disabled when the server gate is off. The UI renders unsupported, unverified, synthetic,
and unavailable states; it does not hide them or recompute scores.

ER4 adds `GET /api/v1/review-reuse/capabilities` with
`{schema_version, decisions_enabled, validated_identity_required, tenant_validated,
reviewer_validated, reviewer_kind, decision_submission_allowed, store_backend}`.
`validated_identity_required` is always true for ledger writes. The endpoint uses existing
authentication, is read-only, never returns key material or tenant inventory, and lets the
UI display gate state without probing the decision POST. The workbench sends the loaded
`task_revision` and canonical `evidence_pack_sha256` with a decision request after ER2; the
server remains the only enforcement authority.

**Re-escalates to L3:** any new auth/session/identity path, client-side ledger or durable
state, decision/default-off semantic change, score/confidence recomputation or ranking,
LLM/assistant surface, canonical write-back, or new export destination.

The UI MUST NOT preselect a human decision, supply a default rationale, place API keys in
browser persistent storage, load external CDN assets during isolated-sample work, or imply
AI/release authority. The read-only capabilities endpoint above and future list pagination
are allowed under this lock, but implementation stays deferred until ER1-ER3 close.
