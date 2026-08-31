# L3 ReviewReuse ER1 + ER2 - Development and Verification

**Date**: 2026-08-29<br>
**Last verification update**: 2026-08-31<br>
**Status**: IMPLEMENTED / FOR REVIEW; not merged, enabled, deployed, or piloted<br>
**PR**: https://github.com/zensgit/cad-ml-platform/pull/584<br>
**Stacked base**: `docs/workbench-board-post-565-20260829`<br>
**Ratified authority**: PR #583 exact head
`9150e06c75721bf086572ed271b68548104e8300`<br>
**Runtime implementation head**:
`444f65f3ed913b0bf07482ad7326a378adf3a435`
(`fix: close ReviewReuse restart and ledger gaps`), on top of strict numeric
repair `721a5214`, calibration-envelope repair `1d0ad7fd`, the original
runtime commit `6cc55841`, and the intervening fail-first hardening chain.

## 1. Authorization boundary

The owner ratified only the named fail-first ER1 + ER2 contract at the exact
authority head above.

Authorized:

- ER1 collision-safe filesystem store, integrity checks, writer lease, crash
  recovery, and legacy migration.
- ER2 input validation, canonical idempotency, revision CAS, decision
  attribution, reviewed-evidence binding, and canonical HTTP/OpenAPI errors.
- Narrow tests, fixtures, operational scripts, and OpenAPI snapshot changes
  required to prove ER1 + ER2.

Not authorized and not performed:

- merge of #583 or #584;
- enabling `REVIEW_REUSE_DECISIONS_ENABLED` in repository or deployment
  configuration;
- deployment, pilot, customer drawing processing, or customer acceptance;
- ER3 real archive/evidence replay;
- ER4 reviewer UI or capabilities endpoint;
- training/feedback ingestion, `eval_integrity_gate`, assistant, cost-cap,
  model-release metric, or canonical PLM write-back changes.

Tests temporarily set the decision flag inside isolated processes to exercise
the gated write path. That is test execution, not deployment enablement.

## 2. Fail-first discipline

The first implementation-branch commit contains exactly the three test files
named by design-lock section 9.1:

| Item | Evidence |
|---|---|
| Test commit | `b708f519b6fba347a2b1b22603ed7fda434fba7e` |
| Parent / authority | `9150e06c75721bf086572ed271b68548104e8300` |
| Files | `test_review_reuse_er1_store_integrity.py`, `test_review_reuse_er2_ledger.py`, `test_review_reuse_api_integrity.py` |
| Baseline result | **18 failed** |
| Runtime-head result | **140 passed** |

Exact command:

```bash
/opt/homebrew/bin/python3.11 -m pytest -q \
  tests/unit/test_review_reuse_er1_store_integrity.py \
  tests/unit/test_review_reuse_er2_ledger.py \
  tests/unit/test_review_reuse_api_integrity.py
```

The red result was reproduced from detached commit `b708f519`, before any
runtime implementation. At `6cc55841`, no baseline named test had been
removed, renamed, skipped, or marked xfail. The current files contain all 18
named tests plus narrower adversarial cases.

## 3. ER1 implementation

### 3.1 Tenant and path integrity

- Literal tenant ids are validated without lossy replacement or truncation.
- Filesystem directories use
  `tenant-v1-<sha256(literal UTF-8 tenant id)>`.
- Every tenant directory carries an exact four-field `tenant.json` sidecar.
- Reads verify directory, sidecar, tenant, task id, record schema, EvidencePack
  identity, and digest ownership.
- Task, candidate, event, and human-decision model envelopes reject unknown
  fields on native reads and legacy migration. Contract-defined dictionary
  payloads remain open only within their named fields.
- Canonical lowercase hyphenated UUID task ids are the only task filenames.
- Reads do not create tenant directories, sidecars, or indexes.
- Lexical traversal, nested symbolic-link traversal, and out-of-root task,
  index, sidecar, and writer-lock paths fail closed.

### 3.2 Store and index integrity

- Task records are the source of truth for create idempotency ownership.
- The exact idempotency-index schema and every owner task are validated.
- Corrupt records and indexes map to `store_record_corrupt` or
  `store_index_corrupt`; stored corruption cannot surface as a request-level
  validation error.
- Missing create-index entries are recovered only under the writer lease by
  scanning identity-valid records.
- One exact matching owner is replayed and indexed; duplicate or mismatched
  owners fail closed.
- List, metrics, store ops, and pilot preflight validate the complete tenant
  record set; audit export validates its requested record and tenant index.
- Full-store validation enumerates each hashed tenant directory and accepts
  only `tenant.json`, optional `idempotency.json`, `tasks/`, and exact internal
  atomic-write temporary-file shapes. Unknown tenant-level files, directories,
  symbolic links, and entry type mismatches fail closed.

### 3.3 Single writer and durable writes

- A write-capable filesystem store holds a non-blocking process-lifetime
  `flock` lease at a stable parent-directory path derived from the canonical
  store root. The lease inode and pathname do not move when migration swaps the
  store root.
- Read-only export and operational validation use explicit read-only stores.
- Mutations reject an inherited/forked or otherwise unavailable writer lease;
  closing an inherited handle cannot unlock the parent process lease.
- Writes use unique same-directory temporary files, file `fsync`, atomic
  replacement, and parent-directory `fsync`.
- Creating a store, staging root, or previously missing ancestor chain fsyncs
  every containing namespace entry before the root can be acknowledged.
- Startup removes only exact internal atomic-write leftovers while unknown
  files and symbolic links continue to fail closed. Read scans ignore only the
  same exact temporary-file shape, so an interrupted write cannot permanently
  block list, metrics, or idempotency recovery.
- Full-store validation enumerates every root entry. It accepts only recognized
  tenant/staging directories and the exact legacy internal `.writer.lock`
  regular file; unexpected files, special files, and symbolic links fail
  writer startup, preflight, and migration.
- A first tenant is assembled under an exact hidden staging name and atomically
  published only after `tasks/` and `tenant.json` are durable. Failed staging
  is removed and a retry starts from an absent tenant rather than a partial
  directory.
- A task is written before its create index. Index write failure marks the
  writer unhealthy, preventing a duplicate task until recovery.
- If a task or first-tenant rename becomes visible but its parent-directory
  `fsync` fails, the current store instance is quarantined. Subsequent reads
  and writes fail with `store_writer_conflict` rather than acknowledging an
  uncertain create or decision through an idempotent retry.
- Writer startup establishes the recovery durability barrier before serving:
  it removes only exact internal leftovers, then `fsync`s recognized task,
  tenant, and root directories deepest-first before full-store validation.

### 3.4 Legacy migration

- Dry-run is the default and validates all records and index ownership before
  data replacement.
- Mixed layouts, path/embedded-identity mismatch, collisions, symbolic links,
  invalid idempotency, and unprovable decisions abort migration.
- Each legacy tenant directory is closed to the recognized `tasks/` directory
  and optional `idempotency.json`; unknown sibling files, directories, links,
  and special entries abort both dry-run and apply before any root swap.
- Apply builds a separately locked staging store, keeps the new-store lease
  held through the rename, and holds the canonical-root lease across the entire
  backup/publish swap. A competing writer cannot create or serve a replacement
  root between the two renames. Apply retains a uniquely named legacy backup
  and restores it after publish or durability failure.
- Backup and successful publish namespace changes are followed by an `fsync`
  of the containing directory. Rollback completes all recovery renames before
  its durability barrier; a persistent rollback `fsync` retains the staged
  migrated copy rather than deleting the last spare.
- If backup restore fails, migration republishes the fully built staging store.
  If both root recovery attempts fail, the legacy backup and migrated staging
  copy remain intact for operator recovery instead of being cleaned up.
- A directory that looks like a new layout is validated before
  `already_migrated` can be returned.
- A write-capable store validates the whole existing layout at startup and
  refuses legacy or mixed directories before any hashed tenant write. The
  explicit dry-run/apply migration remains the only upgrade path.

## 4. ER2 implementation

### 4.1 Input and canonical JSON

- Empty, oversized, and non-DXF inputs are rejected before task persistence.
- Upload reading is bounded in chunks by
  `REVIEW_REUSE_MAX_UPLOAD_BYTES` (default 52,428,800).
- `canonical_json_v1` implements the approved JCS/I-JSON digest contract:
  UTF-16 property ordering, deterministic UTF-8, ECMAScript-compatible
  binary64 formatting, negative zero as zero, and rejection of duplicate keys,
  non-finite values, unsafe integers, lone surrogates, and unsupported values.
- Create and decision digest payloads are independently recalculated by the
  store rather than trusted from the service.

### 4.2 Atomic task and decision ledger

- The first task snapshot is revision 1.
- Every persisted mutation is a store-level CAS and increments revision by
  exactly one.
- Existing events are immutable and every state mutation appends an event.
- The final newly appended event must match the accepted old/new state
  transition. A decision event must also bind state, reviewer, candidate,
  reviewed revision, and EvidencePack digest to the committed decision.
- Native reads, list, migration, and audit export apply the same five-field
  binding to every non-empty decided event ledger. The final event must be
  `decision_submitted`; deleting it, changing its type, or changing any bound
  detail fails as `store_record_corrupt`.
- Ratified legacy decided records with an empty event list remain readable.
  Because the current model carries no native/migrated provenance, ER2 does
  not claim detection when a privileged writer replaces an entire native
  event list with `[]`; closing that ambiguity requires a separate L3 schema
  amendment.
- Immutable task identity, trace, source filename/hash, idempotency ownership,
  and reviewed candidate set cannot be rewritten during cancel or decision.
- Once a task reaches `evidence_ready`, its reviewed candidates and calibration
  version/status are frozen for the decision or cancel commit.
- Concurrent cancel/decision calls permit exactly one commit.
- Pipeline failures retain attempted events, clear invalid candidates/evidence,
  persist a safe failure code/message, and do not emit
  `evidence_pack_ready`.
- Adapter output is checked by the same canonical snapshot validator before
  the ready event or final CAS. Structurally valid but store-invalid candidate
  results therefore close as a failed task instead of stranding `running`.

### 4.3 Decision and identity binding

- Decision submission remains default-off.
- A write requires both a validated tenant and a
  `principal-v1-<sha256({issuer, subject})>` reviewer.
- A successfully signature-verified JWT establishes tenant isolation even
  when optional posture has no configured issuer; reviewer-principal trust
  remains independently issuer-bound and therefore unavailable in that case.
- The active settings class declares JWT audience and issuer, so environment
  values used by boot validation are also visible to the middleware.
- JWT `tenant_id` and `sub` must be exact strings; numeric claims are not
  string-coerced into trusted identities.
- A trusted tenant claim cannot enter the reserved `ak-` fallback namespace.
- Decision-enabled boot requires required integration auth plus JWT secret,
  audience, and issuer, even in development posture.
- Decision-enabled boot rejects a whitespace-padded issuer instead of booting
  into a posture where every reviewer principal later fails validation.
- `expected_revision` is a required strict integer, and the request must carry
  the exact current canonical EvidencePack digest.
- The store independently binds the committed decision to the pre-commit
  revision and EvidencePack digest.
- A digest-valid EvidencePack must bind both nested calibration version and
  status to the owning task; migration refuses mismatches before recomputing a
  digest.
- A persisted EvidencePack must have exactly the builder-defined top-level key
  set, and its schema, source, candidate evidence, confidence, rejection,
  unsupported-state, and provenance envelopes must equal the canonical
  builder output. Re-signing changed or extra fields does not make them valid.
- A persisted EvidencePack's `human_decision` block must exactly match the
  task decision, including the undecided `null` state; recomputing the pack
  digest cannot smuggle a decision into an undecided task.
- Audit export validates and renders the EvidencePack from the same task
  snapshot used for the exported task and event ledger. A concurrent decision
  or cancel cannot create a mixed-revision bundle through a second read.
- Candidate rejection reasons are checked against the ratified closed
  vocabulary before native persistence and legacy migration acceptance.
- A persisted decision must identify exactly the preceding task revision. Its
  reviewed digest must equal the canonical EvidencePack reconstructed for that
  pre-decision `evidence_ready` snapshot, including decisions without an
  idempotency key.
- Migration distinguishes missing legacy calibration metadata from explicit
  JSON `null`: a null calibration object, version, or status is corrupt and is
  never silently backfilled and re-signed.
- Client-supplied derived identity fields are forbidden rather than ignored.
- Candidate/state matrix, closed new-submission reason vocabulary, non-empty
  rationale, actor-bound idempotency, and terminal-state semantics are
  enforced.
- The reason-code vocabulary is shared by the service and store. Native
  `evidence_ready -> decided` CAS mutations revalidate it even when a caller
  bypasses the service; legacy migration continues to preserve historical
  out-of-vocabulary reason codes as required by the ratified contract.

### 4.4 HTTP and OpenAPI contract

- Post-auth framework validation uses
  `{detail: {code, message}}`.
- Existing platform authentication remains a documented 401
  `{detail: string}` boundary.
- Domain 400/403/404/409/413/415/422/500/503 responses are documented for all
  ReviewReuse routes.
- Unexpected domain codes become a sanitized 500 rather than a catch-all 400.
- Duplicate-key rejection applies to `application/json`,
  `application/*+json` request media types including parameters, and a
  non-empty body that FastAPI treats as JSON when `Content-Type` is absent.
- Operator logs receive only the structured store failure code; tenant
  responses contain no path, key, token, or record payload.
- Isolated/pilot/JWT runbooks no longer present API-key fallback decisions,
  stale reason codes, raw-sub attribution, raw-tenant filesystem paths, or a
  request missing the required revision/EvidencePack binding. Decision examples
  require both platform API-key and validated Bearer JWT authentication and
  remain explicitly owner-only.

## 5. Independent-review hardening

The runtime was reviewed again before any merge. Six narrower adversarial cases
were reproduced against the first runtime head and fixed in `1ee1bad0`:

| Finding | Red proof | Closed behavior |
|---|---|---|
| Forked child `close()` released the parent `flock` | 1 focused failure | child closes only its inherited descriptor; parent lease stays exclusive |
| Existing legacy layout was accepted until the first mixed-layout write | included in a 3-failure batch | writer startup refuses legacy/mixed layout before serving |
| Decision-enabled boot accepted padded issuer configuration | included in a 3-failure batch | exact issuer configuration required |
| `calibration_status` accepted values outside the design-lock enum | included in a 3-failure batch | persisted model is restricted to `uncalibrated` or `calibrated` |
| Crash-left atomic temp files permanently broke task scans | included in a 2-failure batch | exact internal leftovers are safely ignored/cleaned under the lease |
| First-tenant sidecar creation could leave a non-retryable half-directory | included in a 2-failure batch | staged tenant tree is atomically published or fully removed |

A second exact-runtime review then found three additional cases. All three were
reproduced together as **3 failed** and fixed in `9494cc24`:

| Finding | Red proof | Closed behavior |
|---|---|---|
| Migration's root-local lease moved into the backup before the new root was published | included in the 3-failure batch; competing writer acquired the recreated root | migration and runtime writers share a canonical-root stable lease acquired before root creation |
| Pilot preflight accepted a whitespace-padded issuer that runtime boot rejects | included in the 3-failure batch; preflight returned 0 | preflight reports issuer exactness and exits 2 |
| Legacy migration re-signed an unknown nested `calibration.status` | included in the 3-failure batch; dry-run succeeded | nested calibration must equal the task's closed status before digest recomputation |

After each red reproduction, the focused cases were rerun green. The three new
cases are **3 passed** and the complete named fail-first command is **40 passed**
at `9494cc24`.

A third exact-head review found three more ER1 integrity gaps. Seven behavioral
cases reproduced the defects as **7 failed** while one explicit legacy-lock
compatibility case already passed. They were fixed in `1bcf275d`:

| Finding | Red proof | Closed behavior |
|---|---|---|
| Migration root renames were not durable across power loss | success and rollback fsync assertions both failed | backup, publish, and rollback renames fsync the store parent |
| Digest-valid nested calibration version could differ from the task | persisted-read and migration cases both succeeded | version and status are bound to the task before acceptance or re-signing |
| Full-store validation ignored unknown root files and dangling symlinks | read validation and file-only migration cases succeeded | every root entry is classified; only tenant/staging directories and exact legacy lock are accepted |

The eight-case focused batch is **8 passed** and the complete named fail-first
command is **48 passed** at `1bcf275d`.

A fourth exact-head GitHub review at `96d1ba4b` found two migration gaps. The
two null-field cases plus persistent rollback-`fsync` case reproduced together
as **3 failed**; a separate backup-restore recovery case reproduced as **1
failed**. The first repair batch made all four pass:

| Finding | Red proof | Closed behavior |
|---|---|---|
| Rollback performed a parent `fsync` while the configured root was absent | persistent `fsync` failure left the legacy root missing | rollback restores or republishes a complete root before its durability barrier |
| Present `calibration.version/status = null` was treated as missing | both malformed packs were accepted and re-signed | key presence is distinguished from absence and explicit null fails closed |
| Backup restore failure could leave the configured root absent | recovery exception left only off-path artifacts | the fully validated staging store is republished as the fallback root |

A read-only Grok 4.6 review then found two narrower durability/schema cases and
one missing recovery-path assertion. The new batch reproduced **2 failed / 3
passed** before the repair. The complete focused migration batch is now **6
passed**:

| Finding | Red proof | Closed behavior |
|---|---|---|
| A failed rollback `fsync` still triggered deletion of the staged spare | restored root existed but the migrated staging copy was removed | staging cleanup is allowed only after the rollback namespace is durably synced |
| Whole-object `calibration = null` differed from live-store validation | migration backfilled and re-signed the malformed object | object null and nested nulls all fail closed; only absent legacy fields may be backfilled |
| Both backup restore and staging republish failure lacked direct coverage | the new assertion already passed against the recovery guard | both complete copies are retained when the filesystem rejects both root recovery renames |

At runtime head `60df0f9f`, the complete named fail-first command is **54
passed** and the full ReviewReuse suite is **148 passed**.

A fifth exact-head GitHub review at `02a39a63` found two persisted-integrity
gaps. Follow-up test-only commit
`b8ebacb760854e225fffeb74eecbfa689f47c25a` reproduced them as **5 failed**;
runtime repair `77cd2969b0f097e54628121e104eabca748a8c65` made all five pass:

| Finding | Red proof | Closed behavior |
|---|---|---|
| An indexed idempotency owner could be replayed while a second valid record owned the same key | indexed replay returned the first owner instead of raising | replay, direct lookup, list, startup, and full-store validation scan all task records and reject duplicate ownership |
| A task candidate could drift from its still digest-valid EvidencePack | state, scores, verification, and provenance tampering produced four accepted reads | all candidate-derived EvidencePack fields are rebuilt from the task and compared before a stored record is accepted |

At runtime head `77cd2969`, the complete named fail-first command is **59
passed** and the full ReviewReuse suite is **153 passed**.

A sixth exact-head GitHub review at `0cfd9b7f` found four remaining durability
and evidence-freeze gaps. Follow-up test-only commit
`d91309df1e7da90fd15575822eff70146a1ba45d` reproduced the five behavioral
cases as **5 failed**; runtime repair
`40b006d43a45fd971ab5ef8059bc427ec5f1c157` made the same command **5 passed**:

| Finding | Red proof | Closed behavior |
|---|---|---|
| A task rename could succeed before its parent-directory `fsync` failed, leaving a visible but not durably acknowledged task | create failed, but the same store could immediately read/replay the visible task | the published-write failure quarantines the store; reads and writes fail closed until writer restart establishes a durability barrier |
| First-tenant publish had the same uncertain namespace window at the store root | create failed, but the visible tenant was accepted by immediate list/retry | the published-tenant failure quarantines the store; startup `fsync`s recognized task, tenant, and root directories before validation and reuse |
| An undecided task accepted an arbitrary digest-valid EvidencePack decision block | forged non-empty `human_decision` loaded successfully | persisted and rebuilt decision blocks must match exactly |
| Decision/cancel CAS could change reviewed calibration metadata | version and status mutations both committed against the old reviewed digest | candidates plus calibration version/status are frozen once the current task is `evidence_ready` |

At runtime head `40b006d4`, the complete named fail-first command is **64
passed** and the full ReviewReuse suite is **158 passed**.

A seventh exact-head GitHub review at
`1a89573f3983253c4ac421f5e44e8fc729dfdf35` found two more persisted-integrity
gaps. Test-only commits `738102645c666d3e404176118d355fcb5434b187`
and `5095f2926eaeec2e82872d9558ae484be73dc74f` reproduced all six behavioral
cases as **6 failed** against the prior runtime. Runtime repair
`2370af34519e64362024c520a38065052628fa83` made the focused batch **6 passed**:

| Finding | Red proof | Closed behavior |
|---|---|---|
| A digest-valid pack could replace `schema_version`, top-level `provenance`, or the exact `source` envelope, and could add unknown top-level fields | four changed/re-signed packs loaded successfully | the pack key set and every builder-derived evidence envelope must exactly match canonical builder output |
| An unkeyed persisted decision could replace `reviewed_revision` or `evidence_pack_sha256`, rebuild the final pack, and pass | both changed/rebuilt decided records loaded successfully | the reviewed revision is exactly the previous revision and its digest must match the reconstructed pre-decision canonical pack |

At runtime head `2370af34`, the complete named fail-first command is **70
passed** and the full ReviewReuse suite is **164 passed**.

This is application-level consistency validation for the store managed by this
service. The EvidencePack digest is an unkeyed SHA-256, not a MAC or external
integrity anchor. A privileged actor that rewrites every mutually checked copy
can evade this class of validation; signatures, WORM storage, or an external
audit ledger would require a separately ratified L3 design and are not claimed
by ER1 + ER2.

An eighth exact-head GitHub review at
`87428565a6aac0292caab0f9f85be76b84a1a368` found an ER1 tenant-directory gap
and an ER2 reason-vocabulary ambiguity:

| Finding | Red proof | Closed or bounded behavior |
|---|---|---|
| Unknown files or directories beside tenant metadata/tasks were ignored | test-only `4a4b05c5f6c1299d0d3df79f5972df7483a3a2ca` reproduced **2 failed** across full-store validation, writer restart, and already-migrated detection | runtime `4cc75feef91e393d851d16f3f0022793d86bb1e0` classifies every tenant-level entry; both cases pass |
| A direct native store CAS could persist a reason code rejected by the API | test-only `d06ae3c05490e648500931e9a00409480bce1a5f` reproduced **1 failed**, while the paired historical-migration compatibility case passed | runtime `d682971b1f87536cd14a9fe2c2940e4d4c0266cf` shares the vocabulary and rejects native CAS bypass; both native rejection and historical preservation pass |

The review's broader load-time proposal cannot be applied without contradicting
ratified design-lock section 5: migrated historical decisions must remain
readable even when their reason code predates the current vocabulary. Native
and migrated decisions have no persisted provenance discriminator today.
Distinguishing a tampered native record from a valid historical record at load
time therefore requires a separately ratified schema amendment such as an
integrity-bound reason-vocabulary version or migration provenance. ER1 + ER2 do
not claim that unratified behavior.

At runtime head `d682971b`, the complete named fail-first command is **74
passed** and the full ReviewReuse suite is **168 passed**.

A ninth exact-head GitHub review at
`b7633aad9a66a0e8d2bbd3cbcc2d1466dcf6874a` found three ER2 consistency and
validation gaps. Test-only commit
`cf26776edd5b27b40b029264343cd931b8afe267` reproduced the five behavioral
cases as **5 failed**; runtime repair
`352bab81c07804c68eb7471ab96a6faeb8fefd7f` made the same command **5 passed**:

| Finding | Red proof | Closed behavior |
|---|---|---|
| Audit export read the task twice and could combine task/events from one revision with EvidencePack/Markdown from another | staged snapshots caused two reads and a mixed revision | export validates and renders from one immutable task snapshot |
| Candidate rejection reasons accepted arbitrary strings in native persistence and legacy migration | create and migration both accepted an unknown reason | store validation rejects reasons outside the `RejectionReason` vocabulary on both paths |
| `application/*+json` bodies bypassed duplicate-key parsing | vendor and merge-patch JSON reached the disabled decision gate instead of failing request validation | all application JSON suffix media types use the strict parser before FastAPI model parsing |

At runtime head `352bab81`, the complete named fail-first command is **78
passed** and the full ReviewReuse suite is **173 passed**.

A parallel read-only FastAPI semantics audit then found that a non-empty body
without `Content-Type` is also parsed as JSON by FastAPI. Test-only commit
`ed0c430223fe6a8b051929e3bc583d57b931a64e` reproduced **1 failed**: a
duplicate-key body reached the decision gate and returned 403 instead of
canonical 422. Runtime repair
`50e49686c02fdc386a1345a7783fe636155171f4` applies the same strict parser to
that implicit-JSON path while leaving empty body and non-JSON media requests
untouched. The focused case is **1 passed**.

At runtime head `50e49686`, the complete named fail-first command is **79
passed** and the full ReviewReuse suite is **174 passed**.

A tenth exact-head GitHub review at
`cc7d1e4e6be0249f6dc6916faa708103927957b9` found four remaining ER1/ER2
integrity gaps. Test-only commit
`68d6943880c3d2efa536dbdbca372d32073b3e12` reproduced the ten behavioral
cases as **10 failed**. A companion active-settings case in test-only commit
`fa6ad4e1c3ff05594a7c4c8375f6abd1188404a6` reproduced separately as **1
failed**. Runtime repair
`dc4879166f58fed38afab91f3f834b70465f7289` made the combined focused command
**11 passed**:

| Finding | Red proof | Closed behavior |
|---|---|---|
| Optional verified JWTs without a configured issuer collapsed distinct signed tenants into the shared API-key namespace | two tenant identity cases failed; the active settings companion also failed to load audience/issuer | tenant trust is separated from reviewer-principal trust, and the active settings class exposes both verifier fields |
| A newly created store root or ancestor chain was not durably linked from its parent | writable-store and migration bootstrap fsync assertions both failed | each created namespace component is followed by an fsync of its containing directory |
| Direct CAS could accept a state transition with a contradictory terminal event or decision detail | wrong event type and wrong reviewer detail both committed | the appended suffix has one transition-closing event in terminal position, with decision attribution bound to the task |
| Legacy migration silently ignored unknown tenant sibling artifacts | file/directory cases passed dry-run and apply | a closed legacy entry allowlist rejects all four cases before any swap |

Existing migration fault-injection assertions were advanced by one parent
fsync to account for the new durable staging-root creation barrier; their
backup, publish, rollback, and spare-preservation semantics remain unchanged.

At runtime head `dc487916`, the complete named fail-first command is **88
passed**, the full ReviewReuse suite is **183 passed**, and the integration
auth/production-identity/pilot-preflight regressions are **46 passed**.

An eleventh exact-head GitHub review at
`eb81fc3aa4fd4104e98e726d900c8cb93af1e926` found that Pydantic's default
extra-field behavior silently discarded unknown persisted fields before ER1
integrity validation. Test-only commit
`c93fc5deede7d51f6f9bd9fc449be53ea267775f` reproduced **8 failed** across
native and legacy task/candidate/event/decision envelopes. Runtime repair
`707dcbf7d9d34c8aa5b3cb0ef20b4b15ab6f227d` closes those four model envelopes
with `extra="forbid"`; the same focused command is **8 passed**. Named
dictionary fields such as event detail, candidate scores/verification/
provenance, and EvidencePack remain governed by their existing content
contracts rather than being recursively closed by this change.

At runtime head `707dcbf7`, the complete named fail-first command is **96
passed**, the full ReviewReuse suite is **191 passed**, and the integration
auth/production-identity/pilot-preflight regressions remain **46 passed**.

A twelfth exact-head GitHub review at
`93df64c041b19d00c78c6a7cc85ee110bc0c1124` found two operator-entrypoint
failures and one persisted event-ledger integrity gap. Test-only commit
`df989064787c4ebf6fb9ea24034f5b6faabbd9ff` reproduced the two script cases as
**2 failed** with `ModuleNotFoundError: src`. Runtime repair
`74ec4f37602aba267ac2d46d35c765f3eb7e6b54` made both direct entrypoints and
their existing script suites pass. Test-only commit
`8b104b96d8e2b275505ccd141f1f7e35124ab418` then reproduced the persisted
ledger cases as **10 failed / 1 passed**; the one passing case was the required
legacy empty-event compatibility lock. Runtime repair
`891f8e9cda3aee506859bafeab982c7e80d8835f` made the full focused batch **11
passed**:

| Finding | Red proof | Closed or bounded behavior |
|---|---|---|
| Pilot preflight failed when Make executed the script by path without `PYTHONPATH` | direct subprocess exited before argument parsing with `ModuleNotFoundError: src` | the script bootstraps the repository root before its `src` import; direct `--help` and the real Make target pass |
| Store backup/list/cleanup had the same direct-script import failure | the store-ops subprocess failed identically | the same narrow bootstrap is applied; direct `--help` and the real read-only list target pass |
| Persisted decided records accepted a removed/wrong terminal event or changed decision detail, allowing audit export of contradictory history | seven load mutations, two migration modes, and audit export all failed their fail-closed assertions; legacy `events=[]` remained readable | every non-empty decided ledger is checked on load/migration/export against `decision_submitted` plus the five ratified binding fields; the all-events-empty native/migrated ambiguity remains owner-gated |

The parallel entrypoint audit also reproduced the same repository-import issue
in `review_reuse_isolated_archive_run.py`. That script is the ER3 isolated
archive path, so it was deliberately not changed under the ER1 + ER2-only
authorization. It remains a required ER3 fail-first item rather than hidden
scope expansion.

The event-ledger repair intentionally does not synthesize migration events or
add an unratified provenance field. A valid migrated decided record may have
`events=[]`; after restart the current schema cannot distinguish it from a
native record whose complete event list was erased. Fully closing that case,
and the earlier native-versus-historical reason-vocabulary ambiguity, requires
one separately ratified record provenance/schema amendment or an external
integrity anchor.

At runtime head `21943856`, the complete named fail-first command is **106
passed**, the full ReviewReuse suite is **203 passed**, the integration
auth/production-identity/pilot-preflight regressions are **47 passed**, and the
two operator script suites are **21 passed**.

A thirteenth exact-head GitHub review at
`f314e63a6ace631467a3ff7b2e76e1b0f4847f46` found that malformed adapter
results could pass model construction but fail only at the final store
mutation, leaving the already-persisted revision-1 task permanently
`running`. Test-only commit
`722b9d3baf6dc9ac3ba74ef3258a94fef8062747` reproduced both an unknown
candidate rejection reason and duplicate candidate id as **2 failed**: each
case left the task running. Runtime repair
`36175b144d75c855ca5f22efc7dbfa504544fe28` reuses the store's canonical task
snapshot validator inside the guarded pipeline, after EvidencePack build but
before `evidence_pack_ready` is emitted or the final CAS is attempted.

Both focused cases are now **2 passed**. Rejected adapter output commits one
revision-2 `failed` task with safe `internal_error`, no candidates, no
EvidencePack, no `evidence_pack_ready` event, and a final `failed` event. A
same-key replay returns that same failed task without invoking recall again.
Final store CAS, writer-lease, and filesystem I/O errors remain outside this
adapter-validation path, so the repair does not retry or overwrite a
concurrent terminal mutation.

Two parallel read-only reviews found no blocking concurrency or idempotency
regression. They confirmed that final-put revision/I/O failures retain their
existing store semantics rather than entering the adapter compensation path.
A dedicated service-level fault-injection test for that final-put boundary is
a non-blocking test-hardening residual; existing CAS and filesystem durability
suites continue to cover the underlying store behavior.

At runtime head `36175b14`, the complete named fail-first command is **108
passed** and the full ReviewReuse suite is **205 passed**.

A fourteenth exact-head GitHub review (`5057891020`) at
`761efd26342e5f274cd3b92746db7b3b4e42f8f5` found two ER2 integrity gaps:
candidate ids could retain surrounding whitespace while decision submission
canonicalized them, and a direct CAS caller could insert an unrelated event
before the otherwise valid transition-closing event. Test-only commit
`ba7b413770196ec500d7d4f260dc8001c059375c` added three assertions and
reproduced the gaps as **3 failed / 4 passed**:

| Finding | Red proof | Closed behavior |
|---|---|---|
| A direct adapter result could persist a noncanonical candidate id | the malformed-adapter case returned `evidence_ready` instead of committing the safe failed snapshot | the persistence boundary rejects empty or surrounding-whitespace candidate ids, so an adapter bypass enters the existing revision-2 failed path |
| The normal seed/live mapping preserved surrounding candidate-id whitespace | both the task and EvidencePack contained `" archive-1 "` | adapter ingress trims the external identifier before model construction; an empty result is rejected rather than invented |
| A decision CAS could append `submitted` before a valid `decision_submitted` | the contradictory event suffix committed successfully | every transition validates its complete appended event sequence; `evidence_ready -> decided` accepts exactly one `decision_submitted`, while `running -> failed` accepts only an ordered pipeline prefix followed by `failed` |

Runtime repair `00618205c09b689820944ba3577c3be1e1e42651`
made the focused batch **7 passed**. The complete named fail-first command is
now **111 passed**, the full ReviewReuse suite is **208 passed**, and the
identity/production/preflight regressions remain **47 passed**.

A fifteenth exact-head GitHub review (`5057978816`) at
`277a9b138d893d71233e6d07e168c9ad64f37ba8` found three remaining bounded
ER1/ER2 review comments covering four gaps. Test-only commit
`783d3996be55d8ce1392d0874a6836e64cee0939` added five behavioral assertions
and reproduced them as **5 failed**:

| Finding | Red proof | Closed behavior |
|---|---|---|
| Strict JSON parsing applied to every ReviewReuse route and buffered an unauthenticated GET body before the platform auth boundary | a body-bearing unauthenticated GET returned canonical 422 instead of the existing platform 401 | only the decision POST receives strict JSON handling; bodyless routes proceed directly to their existing dependency/auth flow |
| The decision JSON route had no pre-buffer body limit | a request larger than 64 KiB reached JSON/model validation and returned 422 | the decision body is read incrementally with a fixed 64 KiB ceiling and returns canonical `input_too_large` / 413 before full buffering |
| Direct native create accepted a non-running revision-1 snapshot or a missing/arbitrary initial event ledger | empty and duplicate-submitted event ledgers both committed | native create requires status `running` and exactly `[submitted, input_validated]`; the separate migrated-record import path retains historical compatibility |
| A valid legacy literal tenant id shaped exactly like a new hashed directory name was misclassified as already migrated | migration treated the legacy directory as a corrupt new layout | new layout classification requires both the hashed directory form and its `tenant.json` sidecar; the sidecar and complete layout are then validated by the existing fail-closed path |

Runtime repair `572f7d04ea6f339f8e715aabb2a2222e4199e485`
made the focused batch **5 passed**. The complete named fail-first command is
now **116 passed**, the full ReviewReuse suite is **213 passed**, and the
identity/production/preflight regressions remain **47 passed**.

A subsequent independent read-only Sol audit at documentation head
`7b7edf948f74deb84f58a6f92137d46cec0d75a8` found that direct native create
could still persist forged initial event metadata despite the new type
sequence check. Test-only commit
`1af3587abdbfd628110c825aad69ea39b81dbd3a` reproduced five cases as **5
failed**: mismatched submitted file name, zero bytes, boolean bytes,
out-of-order events, and an event timestamp after `updated_at`. Runtime repair
`487c1c58dfbd72db96f06eab0e8aa9a684578ab9` binds the submitted file name to
the task, requires a non-boolean positive integer byte count, and enforces
`0 <= created_at <= submitted.ts <= input_validated.ts <= updated_at` for the
native revision-1 snapshot. The focused initial-ledger batch is **7 passed**.

The ratified `TaskEvent.detail` dictionary remains extensible; this repair
validates the named native-create bindings without inventing a closed detail
schema or changing `_import_migrated`. The complete named fail-first command
is now **121 passed**, the full ReviewReuse suite is **218 passed**, and the
identity/production/preflight regressions remain **47 passed**.

A sixteenth exact-head GitHub review (`5058022742`) at
`5ae43575423d627602b300b1aef76753940f5336` found that the persisted
`decision_submitted` event's timestamp was not bound to its decision, prior
event, or task update time. Test-only commit
`e3b86a99d79cfa76688d6114e729c55e2c27d0b1` added three timestamp mutations;
the existing load-time binding batch reproduced them as **3 failed / 7
passed**. Runtime repair
`b11f3cb1f4b5e62501eb49a4836fecf621428025` enforces
`prior_event.ts <= human_decision.ts <= decision_submitted.ts <=
task.updated_at` for every non-empty decided ledger. The same focused batch is
now **10 passed**.

This check uses existing persisted fields and applies on native load,
migration, mutation, and audit-export validation through the shared payload
validator. It does not alter the separately owner-gated all-events-empty
legacy ambiguity. The complete named fail-first command is now **124 passed**,
the full ReviewReuse suite is **221 passed**, and the
identity/production/preflight regressions remain **47 passed**.

A seventeenth independent read-only review at `13055d4966fb3b543d747759ebd237d24a55e452`
found that a digest-valid EvidencePack could add unknown nested calibration
fields and still pass load validation. Test-only commit
`6693458f9c53b100308059b1f8d4bfaab76987fd` added the smuggling case while
retaining the valid version and status; the focused immutable-envelope command
reproduced **1 failed / 4 passed**. Runtime repair
`1d0ad7fddbec746f9fdd906fe147e66543e00dbe` includes `calibration` in the
existing builder-derived envelope equality check. The focused command is now
**5 passed**, the complete named fail-first command is **125 passed**, and the
full ReviewReuse suite is **222 passed**.

An eighteenth exact-head read-only review at
`af72d0ec02b5d2dc1d92508539bc89ba857245a8` found that Pydantic's default
numeric coercion allowed native ledger JSON to replace number-valued task
timestamps, event timestamps, candidate scores, or the human-decision
timestamp with strings. Validation normalized those bytes back to floats and
accepted the corrupt record. Test-only commit
`c0472d95c2b5fd3855db334281587293708cac75` added four persisted-record
mutation cases; the focused command reproduced **4 failed**. Runtime repair
`721a52140857d32fa4927cfda60af5175c7dd030` applies field-level `StrictFloat`
types to the five governed numeric model positions. JSON integers and floats
remain accepted and normalize to floats; strings and booleans fail schema
validation without making legal enum strings globally strict. Final coverage
extends the original red cases across all five positions and both invalid
types. The focused command is now **10 passed**, the complete named fail-first
command is **135 passed**, and the full ReviewReuse suite is **232 passed**.

A nineteenth exact-head review at
`91b19d6491cbce44f935f467800cc9a10a5db56d` found two remaining ER2 runtime
integrity gaps. First, a persisted `evidence_ready` record accepted a changed
middle event type, a regressed event timestamp, or an `updated_at` preceding
its final event. Second, process death after durable revision-1 create and
before final EvidencePack commit left a keyed task permanently `running`; a
same-key retry returned that snapshot without resuming or terminalizing it.

Test-only commit `390daf250697549c6ef5c5f3183e67a4f52cf504`
reproduced the ledger mutations as **3 failed** and the restart case as **1
failed**. Test-only commit
`3446ea6d1452256a684e287a15b086c8c3328504` separately reproduced a regressing
wall clock as **1 failed**: the event generator could violate the new timestamp
invariant before the first durable create.

Runtime repair `444f65f3ed913b0bf07482ad7326a378adf3a435` now validates every
non-empty runtime event ledger against its legal status sequence and monotonic
`created_at -> events -> updated_at` chronology. Event generation uses one
timestamp clamped to the prior task update time. After a writer lease ends, a
new filesystem writer converts each canonical native `running` snapshot to one
revision-2 `failed/internal_error` snapshot before serving; a same-key retry
returns the same terminal task and never creates a second id.

Eventless migrated records remain untouched because the current schema cannot
prove whether `events=[]` is legitimate history or an erased native ledger.
The startup recovery therefore does not manufacture a modern `failed` event on
an unprovable legacy prefix. That boundary remains the separately owner-gated
schema/provenance issue already recorded below.

Independent read-only Grok 4.6 and Kimi K3 reviews corroborated the restart and
event-ledger findings. Kimi also traced the default-off live-dedup hook to a
vision API with no tenant/index namespace; this is the already-deferred ER3
isolated-archive problem, not authority to modify or enable that path in ER1 +
ER2. Its local-filesystem directory-swap race observation requires write access
inside the protected store tree and a separately reviewed dirfd-based design;
it is recorded as residual hardening, not silently folded into this patch.
The read-only Claude Code run exited without a usable report and is not counted
as review evidence.

## 6. Verification evidence

Runtime-affected commands below ran locally with Python 3.11.15 at runtime head
`444f65f3`.

| Gate | Result |
|---|---:|
| Exact named fail-first command, including narrower additions | **140 passed** |
| Focused event-ledger mutation batch | **3 passed** (red: 3 failed at test-only `390daf25`) |
| Focused writer-restart recovery case | **1 passed** (red: 1 failed at test-only `390daf25`) |
| Focused regressing-wall-clock case | **1 passed** (red: 1 failed at test-only `3446ea6d`) |
| Focused strict persisted-numeric batch | **10 passed** (red: 4 failed at test-only `c0472d95`) |
| Focused digest-valid calibration-smuggling batch | **5 passed** (red: 1 failed / 4 passed at test-only `6693458f`) |
| Focused null/rollback/recovery batch | **6 passed** |
| Focused duplicate-owner/candidate-evidence batch | **5 passed** |
| Focused published-write quarantine/decision/calibration batch | **5 passed** (red: 5 failed at test-only `d91309df`) |
| Focused immutable-envelope/reviewed-snapshot batch | **6 passed** (red: 6 failed at test-only `5095f292`) |
| Focused tenant-artifact batch | **2 passed** (red: 2 failed at test-only `4a4b05c5`) |
| Focused native/historical reason batch | **2 passed** (red: 1 failed / 1 passed at test-only `d06ae3c0`) |
| Focused single-snapshot/candidate-vocabulary/JSON-suffix batch | **5 passed** (red: 5 failed at test-only `cf26776e`) |
| Focused implicit-JSON duplicate-key case | **1 passed** (red: 1 failed at test-only `ed0c4302`) |
| Focused identity/event/root/legacy-artifact batch | **11 passed** (red: 10 failed at `68d69438` plus 1 failed at `fa6ad4e1`) |
| Focused unknown record-field batch | **8 passed** (red: 8 failed at test-only `c93fc5de`) |
| Focused operator direct-entrypoint batch | **2 passed** (red: 2 failed at test-only `df989064`); complete script suites **21 passed** |
| Focused persisted decision-event batch | **11 passed** (red: 10 failed / 1 legacy-lock pass at test-only `8b104b96`) |
| Focused malformed-adapter failure batch | **2 passed** (red: 2 failed at test-only `722b9d3b`) |
| Focused candidate/event canonicalization batch | **7 passed** (red: 3 failed / 4 passed at test-only `ba7b4137`) |
| Focused bounded-body/native-ledger/legacy-layout batch | **5 passed** (red: 5 failed at test-only `783d3996`) |
| Focused initial-event metadata batch | **7 passed** (red: 5 failed at test-only `1af3587a`) |
| Focused decision-event timestamp batch | **10 passed** (red: 3 failed / 7 passed at test-only `e3b86a99`) |
| `make PYTHON=/private/tmp/cadml-review-reuse-testenv-20260831/bin/python test-review-reuse` | **237 passed** |
| Integration-auth + production-identity + pilot-preflight regressions | **47 passed** |
| `make PYTHON=/opt/homebrew/bin/python3.11 test-core` | **39 passed** |
| `make PYTHON=/opt/homebrew/bin/python3.11 validate-openapi` | **5 passed** |
| `make PYTHON=/opt/homebrew/bin/python3.11 validate-core-fast` | **exit 0; 229 tests + 10 governance checks** |
| JCS finite-binary64 differential vs Node `JSON.stringify` at `1ee1bad0`; canonical module unchanged at current head | **20,000 cases; 0 mismatches** |
| Black + isort | **pass (4 Python files changed by this hardening)** |
| flake8 | **pass (4 Python files changed by this hardening)** |
| mypy | **success (`store.py`, `service.py`)** |
| `py_compile` | **pass (4 Python files changed by this hardening)** |
| `git diff --check` | **pass** |

The OpenAPI snapshot was regenerated only after explicit development posture:

```bash
ENVIRONMENT=development API_KEY=test \
  /opt/homebrew/bin/python3.11 \
  scripts/ci/generate_openapi_schema_snapshot.py \
  --output config/openapi_schema_snapshot.json
```

Output:

```text
OpenAPI snapshot written: config/openapi_schema_snapshot.json
paths=201 operations=207
```

An earlier attempt without development identity configuration was refused by
the existing production boot gate and wrote no snapshot. That refusal is
expected fail-closed behavior.

Observed warnings are pre-existing `ezdxf`/pyparsing deprecations and
short-key warnings from test-only JWT fixtures. No test failed or was skipped.

A system Python 3.9 attempt is not counted as verification evidence: repository
`conftest` imports uniformly failed because that interpreter cannot evaluate
the project's `str | None` annotations. The isolated Python 3.11 environment
was completed with the repository-pinned `python-multipart==0.0.18` and
`PyJWT[crypto]==2.10.1` dependencies before the successful commands above.

At pushed documentation head
`ee3b6c9eb20189aed7eb068499e9074ec4025421`, every emitted GitHub check
completed without a failing conclusion. Those checks covered action pinning,
metrics, and stress/observability workflows only. Repository workflow search
found no job selecting `make test-review-reuse`; therefore the green GitHub
state is not represented as execution of the 237-test ReviewReuse gate. Draft
PR #581 is unrelated synthetic-superpass CI work and does not close this gap.

## 7. Boundary verification

- The implementation diff has no ER3, ER4, capabilities endpoint,
  `eval_integrity_gate`, training/feedback, assistant, or cost-cap runtime
  changes.
- No repository or deployment configuration enables decisions or live dedup.
- ER1 + ER2 do not claim cryptographic tamper evidence against a privileged
  filesystem writer; that would require a separate owner-ratified L3 design.
- ER2 does not claim load-time native-versus-migrated reason provenance; the
  current ratified schema cannot represent that distinction.
- ER2 rejects missing or changed terminal decision events when the persisted
  ledger is non-empty, but does not claim native-versus-migrated provenance
  for an entirely empty ledger. The exact-head review thread remains open for
  that separately gated schema decision.
- Initial event detail remains an extensible dictionary under the ratified
  contract. Native create binds the required file-name and byte-count fields;
  closing all detail keys would require a separate contract amendment.
- Writer-restart recovery terminalizes only native `running` records with a
  canonical non-empty event prefix. It deliberately leaves eventless migrated
  records unchanged rather than inventing provenance.
- Default-off live dedup still has no tenant/index namespace in its legacy
  vision seam. It remains excluded from pilot authority and must be replaced by
  the isolated ER3 archive path before any real replay.
- ER1 + ER2 do not claim race-free operation against a privileged local actor
  that can rename directories inside the protected store tree. Closing that
  threat requires a separately reviewed dirfd/`O_NOFOLLOW` filesystem design.
- The 64 KiB decision-body ceiling is enforced while the ASGI request stream
  is consumed. Whether a deployment proxy or ASGI server buffers the complete
  body before application code runs remains a deployment-chain verification
  gate, not a claim of this local test suite.
- The direct-entrypoint issue independently observed in the ER3 isolated
  archive script was not repaired under this ER1 + ER2 authorization.
- The canonical checkout remained clean; all work occurred in the isolated
  `codex/review-reuse-er1-er2-20260829` worktree.
- PR #584 is stacked on the exact ratified #583 branch and remains for review.

## 8. Remaining gates

Engineering-local ER1 + ER2 implementation and verification are complete.
This does not close the L3 release process.

Still required:

1. Exact-head emitted CI on the final #584 documentation head, with no claim
   that it ran the local ReviewReuse target.
2. Owner decision to authorize dedicated `test-review-reuse` CI wiring or
   explicitly accept the recorded local-only execution evidence for this PR.
3. Owner review of the amended #584 head and explicit merge authorization.
4. Separate owner authorization before any decision enablement.
5. Separate ER3 implementation authorization and approved isolated data before real archive
   replay.
6. Owner authorization to synchronize #585 onto the final #584 head, followed
   by a new exact-object ER3 design audit; the current #585 base is stale.
7. ER4 remains deferred until ER1-ER3 prerequisites close and the owner opens
   its implementation window.
8. A separate owner-ratified schema/provenance amendment is required before
   load-time native-versus-migrated reason provenance and an entirely erased
   native event ledger can both be distinguished from valid legacy records.
