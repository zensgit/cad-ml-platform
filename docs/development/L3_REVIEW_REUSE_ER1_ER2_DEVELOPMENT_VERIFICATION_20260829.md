# L3 ReviewReuse ER1 + ER2 - Development and Verification

**Date**: 2026-08-29<br>
**Status**: IMPLEMENTED / FOR REVIEW; not merged, enabled, deployed, or piloted<br>
**PR**: https://github.com/zensgit/cad-ml-platform/pull/584<br>
**Stacked base**: `docs/workbench-board-post-565-20260829`<br>
**Ratified authority**: PR #583 exact head
`9150e06c75721bf086572ed271b68548104e8300`<br>
**Runtime implementation head**:
`d682971b1f87536cd14a9fe2c2940e4d4c0266cf`
(`fix: enforce native ReviewReuse decision reasons`), on top of the
original runtime commit `6cc55841` and the earlier hardening chain through
`4cc75fee`.

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
| Runtime-head result | **74 passed** |

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
- Immutable task identity, trace, source filename/hash, idempotency ownership,
  and reviewed candidate set cannot be rewritten during cancel or decision.
- Once a task reaches `evidence_ready`, its reviewed candidates and calibration
  version/status are frozen for the decision or cancel commit.
- Concurrent cancel/decision calls permit exactly one commit.
- Pipeline failures retain attempted events, clear invalid candidates/evidence,
  persist a safe failure code/message, and do not emit
  `evidence_pack_ready`.

### 4.3 Decision and identity binding

- Decision submission remains default-off.
- A write requires both a validated tenant and a
  `principal-v1-<sha256({issuer, subject})>` reviewer.
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

## 6. Verification evidence

Runtime-affected commands below ran locally with Python 3.11.15 at runtime head
`d682971b`.

| Gate | Result |
|---|---:|
| Exact named fail-first command, including narrower additions | **74 passed** |
| Focused null/rollback/recovery batch | **6 passed** |
| Focused duplicate-owner/candidate-evidence batch | **5 passed** |
| Focused published-write quarantine/decision/calibration batch | **5 passed** (red: 5 failed at test-only `d91309df`) |
| Focused immutable-envelope/reviewed-snapshot batch | **6 passed** (red: 6 failed at test-only `5095f292`) |
| Focused tenant-artifact batch | **2 passed** (red: 2 failed at test-only `4a4b05c5`) |
| Focused native/historical reason batch | **2 passed** (red: 1 failed / 1 passed at test-only `d06ae3c0`) |
| `make PYTHON=/opt/homebrew/bin/python3.11 test-review-reuse` | **168 passed** |
| Integration-auth + production-identity + pilot-preflight regressions | **44 passed** |
| `make PYTHON=/opt/homebrew/bin/python3.11 test-core` | **39 passed** |
| `make PYTHON=/opt/homebrew/bin/python3.11 validate-openapi` | **5 passed** |
| `make PYTHON=/opt/homebrew/bin/python3.11 validate-core-fast` | **exit 0; 229 tests + 10 governance checks** |
| JCS finite-binary64 differential vs Node `JSON.stringify` at `1ee1bad0`; canonical module unchanged at current head | **20,000 cases; 0 mismatches** |
| Black + isort | **pass (26 files)** |
| flake8 | **pass (26 files)** |
| mypy | **success (12 source/script files)** |
| compileall | **pass** |
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

## 7. Boundary verification

- The implementation diff has no ER3, ER4, capabilities endpoint,
  `eval_integrity_gate`, training/feedback, assistant, or cost-cap runtime
  changes.
- No repository or deployment configuration enables decisions or live dedup.
- ER1 + ER2 do not claim cryptographic tamper evidence against a privileged
  filesystem writer; that would require a separate owner-ratified L3 design.
- ER2 does not claim load-time native-versus-migrated reason provenance; the
  current ratified schema cannot represent that distinction.
- The canonical checkout remained clean; all work occurred in the isolated
  `codex/review-reuse-er1-er2-20260829` worktree.
- PR #584 is stacked on the exact ratified #583 branch and remains for review.

## 8. Remaining gates

Engineering-local ER1 + ER2 implementation and verification are complete.
This does not close the L3 release process.

Still required:

1. Exact-head CI on the final #584 documentation head.
2. Owner review of the amended #584 head and explicit merge authorization.
3. Separate owner authorization before any decision enablement.
4. Separate ER3 implementation authorization and approved isolated data before real archive
   replay.
5. ER4 remains deferred until ER1-ER3 prerequisites close and the owner opens
   its implementation window.
6. A separate owner-ratified schema amendment is required before load-time
   native-versus-migrated reason provenance can be enforced.
