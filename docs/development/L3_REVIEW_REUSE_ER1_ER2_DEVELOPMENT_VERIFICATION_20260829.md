# L3 ReviewReuse ER1 + ER2 - Development and Verification

**Date**: 2026-08-29  
**Status**: IMPLEMENTED / FOR REVIEW; not merged, enabled, deployed, or piloted  
**PR**: https://github.com/zensgit/cad-ml-platform/pull/584  
**Stacked base**: `docs/workbench-board-post-565-20260829`  
**Ratified authority**: PR #583 exact head
`9150e06c75721bf086572ed271b68548104e8300`  
**Runtime implementation head**:
`1ee1bad01097dfc6dec3c774c6a5f263fe92192e`
(`fix: harden ReviewReuse store crash recovery`), on top of the original
runtime commit `6cc55841`.

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
| Runtime-head result | **33 passed** |

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

### 3.3 Single writer and durable writes

- A write-capable filesystem store holds a non-blocking process-lifetime
  `flock` lease.
- Read-only export and operational validation use explicit read-only stores.
- Mutations reject an inherited/forked or otherwise unavailable writer lease;
  closing an inherited handle cannot unlock the parent process lease.
- Writes use unique same-directory temporary files, file `fsync`, atomic
  replacement, and parent-directory `fsync`.
- Startup removes only exact internal atomic-write leftovers while unknown
  files and symbolic links continue to fail closed. Read scans ignore only the
  same exact temporary-file shape, so an interrupted write cannot permanently
  block list, metrics, or idempotency recovery.
- A first tenant is assembled under an exact hidden staging name and atomically
  published only after `tasks/` and `tenant.json` are durable. Failed staging
  is removed and a retry starts from an absent tenant rather than a partial
  directory.
- A task is written before its create index. Index write failure marks the
  writer unhealthy, preventing a duplicate task until recovery.

### 3.4 Legacy migration

- Dry-run is the default and validates all records and index ownership before
  data replacement.
- Mixed layouts, path/embedded-identity mismatch, collisions, symbolic links,
  invalid idempotency, and unprovable decisions abort migration.
- Apply builds a separately locked staging store, keeps the new-store lease
  held through the rename, retains a uniquely named legacy backup, and restores
  the backup if the final rename fails.
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
- Client-supplied derived identity fields are forbidden rather than ignored.
- Candidate/state matrix, closed new-submission reason vocabulary, non-empty
  rationale, actor-bound idempotency, and terminal-state semantics are
  enforced.

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

The first runtime head was reviewed again before any merge. Six narrower
adversarial cases were reproduced against that head and fixed in `1ee1bad0`:

| Finding | Red proof | Closed behavior |
|---|---|---|
| Forked child `close()` released the parent `flock` | 1 focused failure | child closes only its inherited descriptor; parent lease stays exclusive |
| Existing legacy layout was accepted until the first mixed-layout write | included in a 3-failure batch | writer startup refuses legacy/mixed layout before serving |
| Decision-enabled boot accepted padded issuer configuration | included in a 3-failure batch | exact issuer configuration required |
| `calibration_status` accepted values outside the design-lock enum | included in a 3-failure batch | persisted model is restricted to `uncalibrated` or `calibrated` |
| Crash-left atomic temp files permanently broke task scans | included in a 2-failure batch | exact internal leftovers are safely ignored/cleaned under the lease |
| First-tenant sidecar creation could leave a non-retryable half-directory | included in a 2-failure batch | staged tenant tree is atomically published or fully removed |

After each red reproduction, the focused cases were rerun green. The complete
ER1 store plus production-identity set is **41 passed** at `1ee1bad0`.

## 6. Verification evidence

All commands below ran locally with Python 3.11.15 at runtime head
`1ee1bad0`.

| Gate | Result |
|---|---:|
| Exact named fail-first command, including narrower additions | **38 passed** |
| `make PYTHON=/opt/homebrew/bin/python3.11 test-review-reuse` | **131 passed** |
| Integration-auth + production-identity regressions | **32 passed** |
| `make PYTHON=/opt/homebrew/bin/python3.11 test-core` | **39 passed** |
| `make PYTHON=/opt/homebrew/bin/python3.11 validate-openapi` | **5 passed** |
| `make PYTHON=/opt/homebrew/bin/python3.11 validate-core-fast` | **exit 0; 229 tests + 10 governance checks** |
| JCS finite-binary64 differential vs Node `JSON.stringify` | **20,000 cases; 0 mismatches** |
| Black + isort | **pass (26 files)** |
| flake8 | **pass (26 files)** |
| mypy | **success (12 source files)** |
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
