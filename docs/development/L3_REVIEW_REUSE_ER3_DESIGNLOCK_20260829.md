# ReviewReuse ER3 Real Isolated Archive Design Lock

**Status:** FOR REVIEW / NOT RATIFIED / DOCS ONLY

**Date:** 2026-08-30

**Authority:** `docs/PRODUCT_STRATEGY.md` §3.3 and the owner-ratified #583 exact head
`9150e06c75721bf086572ed271b68548104e8300`

**Runtime base:** open PR #584 exact head
`af72d0ec02b5d2dc1d92508539bc89ba857245a8`

**Runtime authority:** NONE

This document proposes the ER3 contract. It does not authorize implementation,
merge, decision enablement, deployment, pilot activity, or customer drawing use.
The current owner authorization covers ER1+ER2 only.

## 1. Decision requested

Gate-A design-lock ratification must name all seven of the following:

1. the exact implementation base SHA;
2. the exact repository-fixture manifest from §4;
3. the digest-pinned private vision image from §5.2;
4. the static substrate attestation exact path and digest from §5.2;
5. the exact EvidencePack v2 contract path and digest from §5.4;
6. the embedded EvidencePack golden-vector digest from §5.4;
7. the fail-first contract in §7.

Without all seven, even Gate A remains blocked. Ratification is deliberately
split: Gate A authorizes only fail-first test/workflow artifacts; Gate B separately
authorizes mock-backed runtime implementation after the exact Gate-A matrix is
reviewed and must additionally name the accepted candidate head plus every
owner-protected evidence identity in §10; Gate C separately authorizes one
repository-fixture run. No gate in this
document authorizes merge, decision enablement, deployment, pilot use, or customer
data.

## 2. Scope

ER3 closes one engineering claim:

> A ReviewReuse task can use a real private archive index and a separate query,
> produce candidates without `seed_candidates`, preserve the source and limits
> of every score, emit a truthful EvidencePack, and replay the recorded export.

In scope:

- one isolated archive manifest;
- private `dedupcad-vision` health, index-add, rebuild if required, and search;
- one ReviewReuse task created through the live recall path;
- truthful candidate score, verification, rejection, and provenance mapping;
- deterministic EvidencePack/audit export and replay evidence;
- fail-closed CLI and discriminating tests.

Out of scope:

- ER4 reviewer UI;
- human decision enablement or decision submission;
- hosted-provider egress;
- production/shared-tenant storage;
- customer or partner samples;
- training, feedback ingestion, `eval_integrity_gate`, cost cap, assistant,
  PLM write-back, deployment, or pilot work.

## 3. Exact-head observations

These observations were reproduced against runtime base `af72d0ec02b5...`:

1. `scripts/review_reuse_isolated_archive_run.py` defaults to synthetic bytes and
   can inject a synthetic candidate with `--seed-similar`; it does not build a
   real archive index.
2. Direct script execution with `PYTHONPATH` removed fails at the first `src`
   import with `ModuleNotFoundError`.
3. `vision_response_to_hits()` copies `visual_similarity` into both `semantic`
   and `visual`, and falls back from absent geometric evidence to generic
   `similarity`.
4. The default live search explicitly sets `enable_geometric=False`; it never
   emits `vision_only_unverified` for a visual-only hit.
5. EvidencePack provenance contains placeholder values
   `dedup2d-workbench-adapter`, `review-reuse-mvp-0`, and `tenant-archive`.
6. EvidencePack confidence is the maximum of unrelated raw score dimensions,
   even when calibration status is `uncalibrated`.
7. `DedupCadVisionConfig` accepts any configured URL without an ER3-locality
   check, and `index_add_2d()` defaults `upload_to_s3=True`.
8. At review time, `DEDUPCAD_VISION_URL` was unset and
   `http://127.0.0.1:58001/health` was unreachable. No real run has occurred.
9. The vision index API has no tenant or index-namespace argument. Connecting
   ER3 to an existing process could contaminate a shared index and make search
   results depend on unknown prior state.
10. The repository exposes no authoritative vision index count/list/reset
    contract and contains no attestation that names every mutable index path for
    the current image. A zero-result query alone cannot prove an empty index.
11. At pinned vision source `2fc35d60...`, `POST /api/index/add` requires query
    parameter `user_name`; the previous draft named only
    `?upload_to_s3=false` and would receive HTTP 422 before indexing.
12. The current runner accepts an arbitrary `--out` plus ambient
    `REVIEW_REUSE_STORE*`, `REVIEW_REUSE_LIVE_DEDUP`,
    `DEDUPCAD_VISION_URL`, proxy, and Docker context variables. A purported
    isolated run can therefore select an unintended store, provider, output, or
    remote Docker daemon unless ER3 rejects that ambient control plane.
13. A one-time `docker inspect` does not seal a shared daemon against posture
    drift between preflight and later copy/exec operations. The previous draft
    also omitted `/dev/shm` and did not close privileged/capability/device/host-
    namespace/host-bind/socket posture.
14. The previous `ER3ArchiveTaskInput` wording supplied already mapped candidates
    and complete evidence context before the service emitted `recall_started`.
    That would make the persisted event chronology false and made the first
    running snapshot require post-search digests that could not yet exist.

Therefore existing synthetic helper tests are regression evidence only. They do
not close ER3.

## 4. Isolated data contract

### 4.1 Repository fixture archive (the only proposed ER3 data source)

Use only repository-tracked deterministic raster fixtures under
`tests/vision/fixtures/cad_features/`. The pinned vision image's index and search
endpoints pass uploads directly to pyvips, Pillow, and OpenCV; they do not invoke
the separate ezdxf routes. Direct DXF input is therefore not an attested contract
for ER3. The selected PNG fixtures were introduced as synthetic golden CAD-feature
samples and contain no customer data.

The query is a separate logical request and run-scoped file name materialized
from the same tracked bytes as `archive-exact-001`. Only the archive role is sent
to index-add. This gives a deterministic exact-duplicate assertion without
adding a second binary copy to the repository.

Recommended deterministic selection:

| Role | Path relative to fixture root | SHA-256 |
|---|---|---|
| archive | `cad_line.png` | `e32de24b2a39e9ee56932c80fc5f28497c01ff36348ee2cb0e363d750ae2334c` |
| query | logical copy of `cad_line.png` | `e32de24b2a39e9ee56932c80fc5f28497c01ff36348ee2cb0e363d750ae2334c` |
| archive | `cad_circle.png` | `c32bd82c54124d87fe0731f7cc1a05e4d0147d755d9cd98dc41a1aaa047495e6` |
| archive | `cad_arc.png` | `9a9ca171ef968202eb69523eb0db9e8f88009cb0048a0d301dfa87076bb48d7e` |

The exact manifest proposed for ratification is:

```json
{
  "schema_version": "review-reuse-er3-archive-manifest-v1",
  "archive_id": "rr-er3-repository-fixture-v1",
  "archive_manifest_sha256": "7fd1e774429fa5f75ab5728ed3e2a55b1972d18d8629da584bcf37dc1de91acf",
  "source_class": "repository_fixture",
  "retention_class": "test_fixture",
  "customer_data": false,
  "entries": [
    {
      "entry_id": "archive-exact-001",
      "role": "archive",
      "path": "tests/vision/fixtures/cad_features/cad_line.png",
      "runtime_name": "archive-exact-001.png",
      "media_type": "image/png",
      "size_bytes": 235,
      "sha256": "e32de24b2a39e9ee56932c80fc5f28497c01ff36348ee2cb0e363d750ae2334c"
    },
    {
      "entry_id": "archive-control-001",
      "role": "archive",
      "path": "tests/vision/fixtures/cad_features/cad_circle.png",
      "runtime_name": "archive-control-001.png",
      "media_type": "image/png",
      "size_bytes": 368,
      "sha256": "c32bd82c54124d87fe0731f7cc1a05e4d0147d755d9cd98dc41a1aaa047495e6"
    },
    {
      "entry_id": "archive-control-002",
      "role": "archive",
      "path": "tests/vision/fixtures/cad_features/cad_arc.png",
      "runtime_name": "archive-control-002.png",
      "media_type": "image/png",
      "size_bytes": 493,
      "sha256": "9a9ca171ef968202eb69523eb0db9e8f88009cb0048a0d301dfa87076bb48d7e"
    },
    {
      "entry_id": "query-exact-001",
      "role": "query",
      "path": "tests/vision/fixtures/cad_features/cad_line.png",
      "runtime_name": "query-exact-001.png",
      "media_type": "image/png",
      "size_bytes": 235,
      "sha256": "e32de24b2a39e9ee56932c80fc5f28497c01ff36348ee2cb0e363d750ae2334c",
      "expected_relationship": {
        "archive_entry_id": "archive-exact-001",
        "kind": "byte_identical_fixture"
      }
    }
  ]
}
```

The displayed digest was computed with `canonical_sha256()` over this object
after removing `archive_manifest_sha256`. Implementation copies this exact
object into the test fixture; it does not invent the manifest after ratification.

This run proves a real engine path with non-customer fixtures. It is not a
customer archive, product acceptance, model-quality claim, or pilot.

### 4.2 Customer/private archive remains deferred

Customer or partner drawings are not an option under this design lock. They need
a separate design and authorization that replaces paths and file names with
opaque exported identifiers, keeps local path bindings outside exported
manifests, defines source-file redaction in EvidencePack, and names the tenant,
retention/deletion window, operator, and endpoint. They must not be committed,
logged by file name, uploaded as CI artifacts, or used by this draft.

### 4.3 Manifest rules

The repository-fixture manifest is canonical JSON and binds:

- `schema_version`, `archive_id`, and `archive_manifest_sha256`;
- one `query` entry and one or more `archive` entries;
- repository-relative source path, unique run-scoped name, role, media type,
  byte size, and file SHA-256;
- source class fixed to `repository_fixture`;
- retention class fixed to `test_fixture` and `customer_data=false`;
- an optional expected relationship used only for test verification, never as a
  recall candidate or score input.

`archive_manifest_sha256` is computed over canonical JSON after removing that
field itself. Entry IDs and run-scoped names must be unique. Every `runtime_name`
must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}\.png$`, and every source path must
be exactly `tests/vision/fixtures/cad_features/<basename>.png` with no second
path segment after the fixed prefix. Duplicate hashes
inside the archive role fail. A source path and hash may occur once in an archive
entry and once in the query entry only when the query declares that archive as a
byte-identical fixture relationship. The query logical role and run-scoped name
are never sent to index-add. Manifest structure/self-digest, path escape, symlink
escape, missing files, unknown fields, or multiple query entries fail before any
Docker or service call. This metadata pass may inspect names but does not open
fixture files and is not trusted as the later content-open proof. After the
runtime preflight in §5.2, the runner opens the repository root once, walks every
fixed path component with directory-FD-relative
`openat(O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW)`, verifies each descriptor with
`fstat()`, and opens the final file relative to the last trusted directory FD with
`O_RDONLY|O_CLOEXEC|O_NOFOLLOW`. It never performs an `lstat`-then-full-path-open
sequence. Each unique source file is opened exactly once, must be regular, must
match the declared size and the 1,048,576-byte cap, and is read with declared-size
bounded reads plus a one-byte EOF check. The same immutable buffer is hashed and
used for every manifest entry that names that source; no path is reopened.
Ancestor replacement, drift, short/long read, symlink, non-regular input, or
descriptor identity mismatch fails with no fixture transfer.

## 5. Runtime contract

### 5.1 Entrypoint and isolation

- Real mode starts only as an owner-bound absolute Python interpreter with exact
  flags `-I -S` and an absolute script path. A stdlib-only bootstrap runs before
  the first `src` import. It requires `sys.flags.isolated`, `ignore_environment`,
  `no_user_site`, `no_site`, and `safe_path`; rejects every parent environment
  key beginning `PYTHON`; binds `sys.orig_argv`, interpreter realpath/parent-chain/
  SHA-256, the clean repository root, and one owner-bound dependency root; and
  rejects current-directory, script-directory, user-site, zip, `.pth`, or unknown
  entries in the initial import path. Only then does it add the exact repository
  and dependency roots, import `src`, and prove every imported ReviewReuse module
  resolves below the authorized repository. The canonical result is
  `python_control_sha256`, persisted in revision 1 and runtime preflight. A shadow
  `src`, `PYTHONPATH`, `PYTHONHOME`, user-site/`.pth` hook, or import-origin drift
  fails before task, Docker, or fixture activity.
- Before Docker endpoint validation, real mode requires owner-supplied absolute
  `--git-bin`, `--authorized-head`, and `--reviewed-command-sha256` values. The Git
  executable is opened by directory-FD/no-symlink traversal, must be one regular
  executable not group/other writable, and its realpath, parent-chain, device,
  inode, mode, size, and SHA-256 are recorded and revalidated immediately before
  each of exactly two no-shell invocations. Those invocations use the absolute
  binary, `--no-optional-locks`, `-c core.fsmonitor=false`, `-c core.pager=cat`,
  `-C <authorized-repository-root>`, and only `rev-parse --verify HEAD` or
  `status --porcelain=v1 --untracked-files=all --ignored=no`. Their child
  environment is the closed four-key map `GIT_CONFIG_NOSYSTEM=1`,
  `GIT_CONFIG_GLOBAL=/dev/null`, `LANG=C.UTF-8`, and `LC_ALL=C.UTF-8`; no `HOME`,
  `PATH`, hook, fsmonitor, repository-
  supplied executable, or ambient `GIT_*` value is used. The repository root is
  opened by trusted directory FD, `HEAD` must equal the authorized exact head,
  and raw status output must be empty. A canonical repository-control artifact
  binds both exact argv/stream receipts, binary and root identities, and the child
  environment. It hashes the exact `sys.orig_argv` process array, including interpreter
  path and `-I -S`, after removing only the
  `--reviewed-command-sha256` option/value and requires that digest to equal the
  owner-supplied value. A canonical repository-state object binds the hashed
  root realpath, observed/authorized heads, reviewed-command digest, empty-status
  digest, and clean result into the runtime preflight. Mismatch fails before any
  Docker or fixture operation. These arguments are integrity pins only: they do
  not authenticate an owner or prove ratification. Gate C remains an out-of-band
  GitHub/operator governance decision; the runner must never infer authority from
  caller-provided values. A future machine-verifiable signature/authorization
  artifact would require a separate L3 design lock and is not part of ER3.
- A new run uses a fresh filesystem store root, tenant, output directory, and
  idempotency key. `--out` must be an absolute lexically normalized path with an
  existing trusted parent. Every parent component is opened relative to the prior
  directory FD with `O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW`; the chain must be owned by
  root or the trusted runner UID and must not be group/other writable. The retained
  parent FD and its path entry are revalidated before root classification, lease
  access, root creation, and summary publication. Aliases resolving to the same
  parent device/inode and basename therefore derive the same lease entry. Before
  checking or creating `--out`, every mutating ER3 subcommand opens and exclusively
  locks the same directory-FD-relative sibling
  `.<run-root-basename>.review-reuse-er3.lock`; it is a mode-`0600`, owner-matching,
  single-link regular file opened with
  `O_RDWR|O_CREAT|O_CLOEXEC|O_NOFOLLOW` outside the run root. `er3-replay`
  instead opens that already-existing directory entry with
  `O_RDONLY|O_CLOEXEC|O_NOFOLLOW` and never `O_CREAT`; a missing lease fails
  before any run-root read and replay creates no path. The lease path is
  permanent: no ER3 command may unlink, rename, truncate, or replace it. The locked FD and directory
  entry must retain the same device/inode/link/mode/owner identity before every
  lifecycle phase. The run-level lease stays held through classification, store
  export-freeze, inventory, summary publication, and parent-directory fsync.
  `er3-replay` holds a shared lock on that same inode across both inventory passes;
  concurrent replays may share it, but a writer/replay conflict returns
  `archive_run_writer_conflict` before reading mutable run state or invoking Docker.
  `--out` is created mode `0700` through the trusted parent directory FD
  and contains every store/control/export artifact. An existing root is accepted
  only by the closed classifier in §5.4: a complete exact revision-1 inventory may
  resume before side effects, and an exact revision-4 root with a durable success
  intent may be finalized without Docker. A partial pre-revision-1 root is
  `archive_run_initialization_incomplete`, never an ordinary resume. Revision 2/3,
  completed, mismatched, linked, extra-artifact, or unknown roots fail with their
  dedicated recovery/finalize/corruption reason; they never start a new run or
  re-execute drawing effects.
- Gate C starts the interpreter from a closed launcher environment containing only
  the explicitly frozen locale and runner-home keys; any `LD_*`, `DYLD_*`,
  `GIT_*`, `PYTHON*`, proxy, Docker, ReviewReuse store, decision, or provider key
  is absent. The OS loader, absolute owner-bound interpreter and Git/Docker
  executables, and approved same-UID runner are the trusted computing base; this
  contract does not claim resistance to a pre-launch compromise of that base.
  Real mode additionally rejects ambient `REVIEW_REUSE_STORE`, `REVIEW_REUSE_STORE_DIR`,
  `REVIEW_REUSE_LIVE_DEDUP`, `REVIEW_REUSE_DECISIONS_ENABLED`,
  `DEDUPCAD_VISION_URL`, all upper/lower-case proxy
  variables, and `DOCKER_HOST`, `DOCKER_CONTEXT`, `DOCKER_TLS_VERIFY`,
  `DOCKER_CERT_PATH`, `DOCKER_CONFIG`, `DOCKER_API_VERSION`, and
  `DOCKER_DEFAULT_PLATFORM`, and every environment key beginning `PYTHON`, `GIT_`,
  `LD_`, or `DYLD_`.
  Docker subprocesses receive a runner-created empty
  mode-`0700` config directory below the run root and the exact owner-bound Engine
  API version; they never inherit the caller's Docker config, credentials,
  context, proxy, or platform selection. It constructs the store
  and archive controller explicitly and never calls an environment-selected
  store or live-dedup factory.
- Presence of `REVIEW_REUSE_DECISIONS_ENABLED` in the parent environment is a
  hard `archive_host_environment_invalid` failure; it is never silently removed.
  The controlled child environment never sets it, and no decision endpoint is called.
- Real mode rejects `--seed-similar`, non-empty `seed_candidates`, and synthetic
  fallback success.

### 5.2 Network and index

- ER3 uses a newly created, dedicated vision container with an empty
  ephemeral data directory. An existing developer, staging, production, shared,
  or customer-populated vision service is forbidden.
- The owner-approved vision image is pinned by digest. Static OCI metadata records
  the expected Docker image ID for each supported platform; a later runtime
  preflight must record and match the actual selected platform and image ID. The
  repository's current CI supplies the image/config candidate, not the isolation
  contract: ER3 additionally requires Docker `--network none`, no host TCP or
  Unix-socket publication, fixed-argv `docker exec` control, explicit ephemeral
  data mounts, and verified cleanup. The service runs with
  `S3_ENABLED=false`, `EVENT_BUS_ENABLED=false`, `EVENT_BUS_USE_LOCAL=true`,
  `ML_PLATFORM_ENABLED=false`, `GEOMETRIC_ENABLED=false`, and `OTEL_ENABLED=false`.
  §5.3 therefore requires visual-only disclosure.
- The current CI candidate is
  `ghcr.io/zensgit/dedupcad-vision@sha256:9f7f567e3b0c1c882f9a363f1b1cb095d30d9e9b184e582d6b19ec7446a86251`.
  This draft does not approve pulling or running it. Any approved pull occurs
  before drawings are mounted or read; the digest is verified before processing.
- The image digest alone is insufficient. The proposed static attestation is
  `docs/development/L3_REVIEW_REUSE_ER3_VISION_SUBSTRATE_ATTESTATION_20260829.json`
  with canonical digest
  `4707f793f3597ebe12da4c6474bbb25208ccd9b1de48920a27c5ca083babe6cd`.
  Its reviewed raw-file SHA-256 is
  `5d7e15aa82d714ad371e106deb3c6310cfc82c3062731392bfb3a6c2db175d94`.
  It binds the OCI index to source revision
  `2fc35d60ff034c9f790868c02381a9716becc942`, records both platform manifests and
  expected image IDs, enumerates search-affecting state, and identifies the
  authoritative count and receipt mechanism. Its status is intentionally
  `static_verified_runtime_unverified`.
- Static source/OCI verification and its explicit runtime gaps are recorded in
  `docs/development/L3_REVIEW_REUSE_ER3_SUBSTRATE_STATIC_VERIFICATION_20260829.md`.
- Static attestation is sufficient only as input to Gate A's fail-first artifacts
  and Gate B's mock-backed implementation. It does not prove tmpfs, empty runtime
  state, network isolation, selected platform, or cleanup. Gate C at the reviewed
  implementation exact head is required before
  image pull/start or fixture-byte processing. The resulting runtime preflight
  receipt must be captured after container start and before any fixture byte is
  opened, transferred, or processed.
- The attestation schema is
  `review-reuse-er3-vision-substrate-attestation-v1` and contains at minimum:
  image reference/digest/expected platform IDs; source repository revision;
  declared and discovered
  mutable database/index/cache paths including private `/dev/shm`; the
  env/config binding and `tmpfs` target
  for each path; disabled integration flags; the authoritative pre/post indexed-
  drawing count plus receipt mechanism and response schema; expected zero and
  post-index counts; network/read-only/control/cleanup posture; and
  `attestation_sha256`. Additional fields are also contract-bound by the
  canonical digest; changing, adding, or removing any field requires a new
  design-lock head and owner response. Its digest excludes only
  `attestation_sha256` itself.
- The container command is overridden to bind the service only to
  `127.0.0.1:8000` inside the container. No port or socket is published to the
  host. The fixed service has no authentication on index/rebuild routes, so a
  host-loopback port is not an isolation boundary and is forbidden.
- Before image inspection, real mode requires `DOCKER_HOST`, `DOCKER_CONTEXT`,
  `DOCKER_TLS_VERIFY`, `DOCKER_CERT_PATH`, `DOCKER_CONFIG`,
  `DOCKER_API_VERSION`, and `DOCKER_DEFAULT_PLATFORM` to be absent. Gate C must
  explicitly approve `--docker-host <absolute-unix-uri>`,
  `--docker-api-version 1.43`, `--platform <linux/amd64|linux/arm64>`, and the
  reviewed command digest; no active/default context is consulted. TCP, HTTP(S),
  SSH, npipe, user-info, query/fragment, percent-encoded, relative, or otherwise
  remote/ambiguous endpoints fail. The URI's absolute socket path is walked by
  directory FD with no symlink component, must be a Unix socket, and its
  canonical path, parent-chain digest, `st_dev`, `st_ino`, owner UID, group GID,
  and permission bits are bound into a `socket_identity_sha256` named by Gate C.
  The parent chain and socket identity are rechecked before every Docker command.
  A mutable path component writable by a principal outside the owner-approved
  Docker-administrator set fails. This narrows the trusted administrative boundary
  to the exact owner-bound local daemon; ER3 makes no claim against its approved
  administrators.
- Gate C also binds `--run-id rr-er3-<32-lowercase-hex>` and an independent
  `--resource-owner-id <64-lowercase-hex>`. Before pull/create, three fixed label-
  filtered Docker list commands must prove zero matching containers, volumes, and
  networks. Together with version, pull, image inspect, create, start, and initial
  inspect, these form exactly nine setup command receipts. The contract contains
  the exact argv matrix for all nine; `golden-setup` is not a live command or an
  authorization shortcut. Every created resource carries both labels. Before each
  of all 49 successful-run Docker commands (9 setup, 12 operations, 24 inspections,
  4 cleanup), a candidate journal is created exclusively at
  `state/.run-journal.next`, fully written and file-fsynced, validated as exactly
  one legal successor, atomically replaced over `state/run-journal.json`, and the
  `state` directory is fsynced. The authoritative journal is never truncated or
  rewritten in place. It records the operation, exact argv digest, class-local
  sequence, and `prepared` stage before spawn. Subprocess result and fresh control
  identities are journaled before receipt artifacts, then the full artifact group
  is fsynced before the journal marks it complete. A zero-exit non-final cleanup
  command leaves `cleanup_status` null. For a nonzero current command, or for the
  zero-exit final branch command after all final resource results are known,
  appending its receipt, clearing `pending_command`, setting phase
  `cleanup_verified`, and setting `cleanup_status` to `failed` or
  `verified_absent` are one atomic validated successor. A non-empty final owned-
  resource result selects `failed`; only complete absence selects
  `verified_absent`. Here `cleanup_verified` means terminal cleanup evidence was
  evaluated, while `cleanup_status` carries the outcome. There is no accepted
  terminal receipt/status-null window. On restart, a bounded incomplete
  candidate is discarded under the run lease, an exact complete successor is
  promoted, and any other candidate/current combination fails without Docker. This
  journal plus the ordered revision-2 prepare and revision-3 command/derived-
  artifact prefixes is the only accepted partial-root grammar. Before recovery
  cleanup begins, one journal-only successor preserves any normal pending command
  byte-for-byte as immutable `interrupted_command` and clears the active pending
  slot. Recovery is then single-attempt: a durable container ID selects the fixed
  four-command remove/container-list/volume-list/network-list branch; a possibly
  successful create without a durable ID starts with the fixed dual-label discovery
  and selects either a three-command zero-match branch or a five-command exact-one-
  match branch. Before create could have run, the no-ID branch is instead exactly
  three non-destructive resource lists and never deletes. A match in that pre-
  create branch, more than one post-create match, malformed evidence, or any
  partial/uncertain cleanup fails as
  `archive_run_recovery_evidence_incomplete` with no further automated Docker
  command. A fully durable current-command result with a nonzero exit, or an exact
  final zero-exit branch with a non-empty owned-resource result, atomically closes
  `cleanup_status="failed"` and returns `archive_cleanup_failed`. Only a grammar-
  accepted complete verified-absent
  branch may re-enter, solely to perform the failed-task CAS with zero Docker.
  Complete ordinary-success and recovery cleanup are intentionally one fail-closed
  equivalence class for a still-running revision-2/3 task. Every branch uses `receipts/cleanup`; no
  parallel recovery-discovery namespace or unjournaled command exists.
- The required absolute `--docker-bin` source is resolved through the same
  directory-FD/no-symlink policy, including its complete parent chain, to a
  regular executable rejected if group/other writable. Its canonical realpath,
  parent-chain digest, and SHA-256 must equal Gate C. Before revision 1, the runner
  copies bytes from that already-open descriptor to fixed
  `<run-root>/control/docker` with exclusive create, mode `0500`, file/directory
  `fsync`, then reopens it relative to the retained mode-`0700` control-directory
  FD and binds device/inode/size/hash. The source path is never executed. Every
  Docker command revalidates and executes only this private copy; its parent is
  writable only by the trusted runner UID, and same-UID compromise is outside the
  boundary. Source/private rename, inode, mode, size, or hash drift fails before
  spawn. Every Docker command receipt retains the freshly observed private-binary
  and socket identity digests; a one-time preflight identity cannot substitute.
  Every command uses the private path as `argv[0]` and `--host
  <owner-bound-endpoint>` as the next two arguments. The child environment has
  exactly five keys and its complete canonical preimage is retained at
  `receipts/preflight/docker-environment.json`; it sets only the controlled empty
  `DOCKER_CONFIG` and exact `DOCKER_API_VERSION=1.43` additions and never uses an
  ambient context, PATH lookup, credential helper, or shell. The canonical
  `docker_control_sha256` binds the source/private binary identities,
  endpoint/socket identity, API version,
  controlled config-directory digest, selected platform, server version, and
  exact parent-environment quarantine. The real-run
  control transport is fixed-argv `docker exec` through that verified
  daemon. The runner never uses `shell=True`, never accepts a
  container name, path, URL, header, form field, retry/redirect option, Unix-
  socket option, or command fragment from the manifest, and validates every
  multipart filename against the manifest's unique `runtime_name`. Mock HTTP
  transports remain test-only and use `trust_env=False` with no retries.
- The runtime preflight verifies the attested paths against image/runtime
  inspection; maps `/app/data`, `/app/indexes`, `/app/logs`, and `/tmp` to
  run-scoped `tmpfs`; proves `/dev/shm` is a private non-host-bound tmpfs; runs the
  container filesystem read-only; mounts no host data path; labels the container
  with the run ID; and records `docker inspect` evidence. Explicit tmpfs overrides
  are required for all three image-declared volumes so Docker cannot create
  anonymous data volumes. An undeclared mutable search path is a hard failure.
  Docker Engine API is fixed to v1.43. The complete raw
  `docker inspect[0].HostConfig` object is retained, strict-I-JSON parsed, and
  checked against the attestation's exact v1.43 source-key set. Every source key
  is either mapped to one normalized field or listed with an exact neutral value;
  unknown, missing, duplicate, or drifted raw keys fail. Its canonical digest is
  included in every strict inspect projection. The normalized HostConfig must
  equal the attestation's closed projection. In addition to the fields below, the projection
  binds auto-remove, log/volume driver, volumes-from, links, supplemental groups,
  DNS fields, cgroup namespace/name/parent, runtime, init, OOM-kill policy, PID
  limit, device cgroup rules, sysctls, storage options, exact tmpfs target/options,
  and shm size:
  network `none`, read-only root, non-privileged, `CapAdd=[]`, `CapDrop=[ALL]`,
  `no-new-privileges`, private IPC, no host PID/UTS/user namespace, devices,
  device requests, binds, port bindings, published ports, extra hosts, or restart
  policy. No additional network attachment or published TCP/Unix socket is
  accepted.
- The complete raw `docker inspect[0].Config.Env` array is separately retained and
  strict-checked against the attestation's exact ordered image environment plus
  nine fixed runtime overrides. Every item is one `NAME=value` string, names are
  unique, and unknown, duplicate, missing, reordered, or drifted entries fail.
  Secrets and all upper/lower-case proxy variables must be absent. Its canonical
  `raw_config_env_sha256` is included in the initial and every fresh inspect
  projection; the 17-field normalized integration/absence view is derived only
  after the raw array passes.
- Every Docker control operation after endpoint resolution, including image
  inspection, create, start, `exec`, cleanup, and labeled-resource listing,
  has a closed ten-field command receipt with exact argv, controlled-environment
  digest, freshly observed private-binary/socket identity digests, integer exit
  code, and raw stdin/stdout/stderr SHA-256. The contract freezes the real setup
  and all 12 operation argv matrices plus every dynamic value source; synthetic
  serialization streams never authorize a command. Every one of the 12
  `docker exec` operations is additionally wrapped by freshly executed before/after normalized `docker
  inspect` commands. Each inspection receipt binds a globally unique observation
  sequence, exact inspect argv and stream digests, and the canonical strict-
  projection digest. A projection digest may repeat when posture is unchanged,
  but an observation receipt or sequence may never be reused. The strict
  projection binds image ID, command
  argv, `/app` working directory, process user, container ID/label, HostConfig,
  mounts, running state, start timestamp, restart count, network attachments,
  published ports, and every attested integration/credential/proxy value or
  absence marker. Drift aborts before the next operation.
  The ordered command/inspection receipts, transferred-content hashes, and
  command counts are hashed into `runtime_seal_receipt_sha256`; the one-time
  preflight digest or a cached inspection cannot substitute for this continuous
  seal. A nonzero Docker exit is rejected except the exact egress-probe `exec`
  exit 7 required below.
- Before any fixture byte is opened, fixed argv first requires
  `/usr/bin/curl --disable --version`
  to exit 0. The no-egress probe does not reuse the service-request common prefix.
  Its sole in-container argv is `/usr/bin/curl --disable --noproxy '*' --proto =http --silent --show-error
  --output /dev/null --write-out '%{http_code}' --connect-timeout 2 --max-time 3
  --max-redirs 0 http://198.51.100.1:80/`. The probe must exit immediately with curl error 7
  and stdout exactly `000`; timeout exit 28, exit 127, or any HTTP response is a
  hard failure. The probe is defense-in-depth only: `docker inspect` network mode
  `none` and absence of attachments remain the authoritative isolation proof.
  `198.51.100.1` is the fixed TEST-NET-2 numeric probe target; DNS and a second
  probe image are not used.
- `GET /health` through `docker exec` to in-container loopback must succeed before
  index mutation and its response is recorded after secret-safe field filtering.
- Health, stats, index-add, rebuild, and both search requests concatenate only the
  attestation's exact curl prefix and exact suffix. The prefix pins numeric
  loopback HTTP, `--noproxy '*'`, `--proto =http`, bounded connect/total time,
  fail-with-body, a final status write-out, and zero redirects. The runner splits
  the final line, requires status exactly 200, and strictly parses only the body;
  redirects, retries, Unix sockets, ambient proxies, user headers/forms/URLs, or
  alternate argument ordering are forbidden.
- Before index-add, `GET /api/stats` at JSON pointer
  `/stats/total_drawings` must report zero. The runner also searches the approved
  query and requires zero candidates, but that query is supplementary evidence
  and never substitutes for the authoritative count proof. The pre-index request
  receives the already verified query bytes directly over `docker exec -i` stdin;
  it creates no container path.
- The service caches search responses for five minutes using file MD5 plus mode,
  result limit, diff, ML, and geometry options. The pre-index probe and post-index
  product query use the exact tuples `(fast, 1, false, false, false)` and
  `(balanced, 5, false, false, true)` respectively, ordered as mode, max results,
  compute diff, ML, and geometry. The runner records both tuples and asserts their
  derived cache keys differ. Reusing the same tuple would permit a stale
  zero-result cache hit and is a hard failure.
- After preflight only, the runner opens each unique validated source through the
  directory-FD chain, verifies size/hash, and retains immutable bytes for the run.
  It writes `inputs/verified-fixture-set.json` as canonical JSON in manifest-entry
  order, binding all four entry IDs/roles/runtime names, all three unique source
  paths, observed sizes/hashes, the manifest digest, entry count, and unique-source
  count. Its self-excluding `verified_fixture_set_sha256` is durably bound by the
  revision-3 event before any upload. Exactly five operations use fixed
  `docker exec -i` argv and pass those immutable verified bytes as raw stdin to
  `/usr/bin/curl --disable --form
  file=@-;filename=<manifest-runtime-name>;type=image/png`; the filename and media
  type are fixed by the manifest and the remaining form/query fields are closed.
  Curl's stdin form buffers the upload to determine its size as documented by the
  [curl command-line reference](https://curl.se/docs/manpage.html). The runner persists
  the exact stdin bytes beside each command receipt and requires
  `stdin_sha256`, `stdin_size_bytes`, and `transferred_content_sha256` to equal
  the verified manifest content. A short, trailing, hash/size-drifted, filename-
  drifted, or reordered argv/input pair fails closed. No source path is reopened,
  and no `docker cp`, tar stream, in-container writer, staged file, or remove
  command exists. The other seven operations bind the zero-byte stdin digest and
  size. Replay revalidates all fixed argv and raw inputs without invoking Docker.
  All 12 operations run as the attested `appuser`; no operation admits `--user`,
  root execution, shell, arbitrary path, or caller-provided code. The existing
  `CapDrop=[ALL]`, no-new-privileges, network-none, no-host-mount/device/socket,
  read-only-root posture and fresh before/after inspection remain mandatory.
  Each archive entry is sent once directly to
  `POST /api/index/add?user_name=review-reuse-er3-fixture&upload_to_s3=false`,
  with its unique manifest-bound multipart filename. Both query parameters and
  their values are fixed because `user_name` is required by source revision
  `2fc35d60...`; omission, duplication, reordering, or substitution fails before
  accepting a receipt. The query entry is never sent to index-add.
- Every external Docker/service JSON artifact first passes an ER3 lossless-number
  lexical profile before ordinary strict parsing: integers remain within the
  I-JSON safe range; each decimal/exponent token is parsed as `Decimal`, converted
  to binary64, canonicalized as one JSON number, and accepted only when that
  canonical number has the same exact decimal value. Underflow, overflow,
  non-finite values, or precision loss such as `1e-400` and
  `9007199254740993.0` fail. Duplicate keys and lone surrogates remain forbidden.
  This check applies again during replay, so the existing float-based
  `strict_json_loads()` is never the sole external trust boundary.
- Each index response is strict I-JSON under that profile with exactly `success`, `drawing_id`,
  `file_hash`, `message`, `processing_time_ms`, and `s3_key`. The runner first
  hashes that complete six-field response, then builds one normalized receipt
  containing manifest `archive_entry_id`, base-10 string `drawing_id`, file hash,
  success, raw-response digest, and its own canonical digest. The receipt digest
  excludes only its self-digest field.
- The receipt set contains the complete normalized receipts in exact manifest
  archive order `archive-exact-001`, `archive-control-001`,
  `archive-control-002`. Its digest excludes only the set self-digest. Neither
  response timing nor provider drawing ID may reorder the set. The exact schema,
  digest preimages, ordering, and golden vectors live in the v2 contract from
  §5.4; ad hoc hashing of selected fields is forbidden.
- Index rebuild through `POST /api/v2/index/rebuild` is required and its response
  recorded. With the event bus disabled, index-add persists storage records but
  does not populate the in-memory pHash and FAISS indexes.
- After index/rebuild, `/stats/total_drawings` must report exactly three. The
  complete set of the three successful index-add receipts must bind distinct
  service-side drawing IDs to exactly the three archive hashes in §4. The service
  has no drawing-list endpoint at this revision; the fresh zero-count precondition,
  exact receipt set, and post-count three form the authoritative identity proof.
  Any mismatch fails before the product query.
- Post-rebuild `/health` must also report both `/indexes/l1_phash/size=3` and
  `/indexes/l2_faiss/size=3`. These are required readiness checks, but they do not
  replace the storage count and receipt set as the archive identity proof.
- The post-index ReviewReuse search runs only after all archive receipts succeed.
  It sends the same immutable query buffer directly once more over
  `docker exec -i` with geometric verification requested; the pre-index and
  post-index uploads are distinct command/inspection receipt groups. Thus each
  archive byte stream is uploaded/indexed once and the query byte stream is
  uploaded/searched exactly twice. Those command counts plus each stdin content
  SHA-256 are bound by the runtime seal receipt. Both query uploads use the same
  manifest-verified bytes; a post-validation source mutation cannot change the
  searched payload.
- A service-level `success=true` is insufficient. Every returned candidate must
  match one distinct `(drawing_id, file_hash, index_receipt_sha256)` triple from
  the canonical receipt set; unknown, duplicate, or digest-mismatched identities
  fail. The result must contain the exact triple bound to `archive-exact-001`,
  and its file hash must equal the query hash. A
  zero-hit response, including `success=true` with empty buckets, fails with
  `archive_recall_incomplete`. This is required because the fixed engine can mask
  an L1 failure as a successful empty response.
- Health, index, rebuild, or search failure produces a non-zero exit and a
  structured failure record. It must not produce a success bundle through the
  ordinary offline `insufficient_evidence` fallback.
- No hosted LLM, external object store, training API, or other egress is allowed.
- The dedicated service is stopped with `docker rm -fv`, then every run-labeled
  container, volume, and network is proven absent. No run network is expected;
  its presence is itself a failure. Cleanup failure emits
  `archive_cleanup_failed`, produces no success bundle, and prevents ER3 closeout.

### 5.3 Score and verification semantics

- For this fixed service contract, `visual` is exactly the finite numeric
  `levels.l2.feature_similarity` in `[0,1]`, with normalization identifier
  `dedupcad-vision-l2-cosine-v1`. An absent, non-finite, or out-of-range value is
  not clamped or replaced; it fails score-source validation for that candidate.
- The complete canonical mapping preimage is embedded in the v2 contract as
  `review-reuse-er3-score-map-v1`, digest
  `ccee15b504054a1cd3def3f6531babbc41a72742d3d2dbb8e35efd57337afd11`.
  For this visual-only run, a receipt-bound candidate with visual score at least
  `0.8` maps conservatively to product state/verdict `similar`, status
  `unverified`, and reason `vision_only_unverified`. Supplier `duplicates` versus
  `similar` bucket, supplier verdict, top-level similarity, and confidence never
  determine the product state. Without deterministic geometry this ruleset never
  emits product state `duplicate`; invalid or below-minimum evidence fails rather
  than being silently relabeled.
- `levels.l1.phash_distance` and `levels.l1.similarity` are retained as raw visual
  method evidence, not copied into `visual` and not averaged with L2.
- `semantic` is exactly `levels.l3.semantic_similarity` only when L3 is evidenced.
  It is `null` in the approved run because ML is disabled.
- `geometric` is exactly `levels.l4.geometric_similarity` only when L4 is
  evidenced. It is `null` in the approved run because geometry is disabled.
- Top-level provider fields `similarity` and `confidence` are supplier aggregates.
  They remain in the strict search-response receipt but are not mapped to any
  normalized score, aggregate confidence, calibration field, or verification
  result.
- If geometry was requested but not executed or not evidenced, geometric remains
  `null` and the candidate carries `vision_only_unverified`.
- Missing dimensions remain JSON `null`, never `0`, copied values, inferred
  values, or fabricated defaults.
- Verification methods may include `vision-l1-phash` only when L1 distance and
  similarity are valid, and `vision-l2-faiss` only when L2 feature similarity is
  valid. A method name is not inferred from a result bucket, top-level verdict,
  `match_level`, or generic similarity.
- An uncalibrated run has `calibration.status="uncalibrated"`, a null calibration
  version/digest, and null aggregate confidence/band. Raw component scores are
  not relabeled as calibrated confidence.

### 5.4 Versioned EvidencePack compatibility

ER3 cannot change the existing `evidence-pack-v1` builder in place. Store loads
and decided-task validation reconstruct that pack; changing its score,
confidence, calibration, or provenance semantics would invalidate ER1/ER2
records and reviewed digests.

The proposed implementation adds a version-dispatched `evidence-pack-v2` path
with one immutable selector:

- Public `POST /api/v1/review-reuse/tasks` remains a v1-only DXF entrypoint and
  continues to return 415 for PNG. The internal runner calls the non-HTTP
  `ReviewReuseService.create_er3_archive_task()` seam with a strict
  `ER3ArchiveTaskInput`. That object accepts only the manifest-bound query PNG
  path/reference plus declared size and digest metadata, ratified manifest/run
  bindings, and one internal archive-controller callable. It does not carry or
  open PNG bytes before revision 2. It also does **not** accept precomputed
  candidates, search responses, or a completed evidence context. It rejects seed
  candidates and never renames PNG bytes to `.dxf`.
- `create_er3_archive_task()` and public `create_task()` share validation and
  validation rules, but v2 uses the cancellation/recovery semantics below and adds
  two explicit `running -> running` CAS transitions. The legal successful v2
  sequence is closed as follows:
  1. revision 1 is `running` with `[submitted, input_validated]`, the immutable v2
     selector, an immutable `ER3RunBinding`, and null evidence context.
     `ER3RunBinding` contains exactly `schema_version`, `run_id`,
     `resource_owner_id`, `archive_manifest_sha256`, `authorized_head`,
     `reviewed_command_sha256`, `python_control_sha256`,
     `docker_source_parent_chain_sha256`, `docker_source_binary_sha256`,
     `docker_private_binary_sha256`, `docker_private_binary_identity_sha256`,
     `docker_socket_parent_chain_sha256`, `docker_socket_identity_sha256`,
     `docker_host`, `docker_api_version`, `selected_platform`, and
     `runner_version`; every value comes from manifest metadata, stdlib bootstrap,
     or the exact Gate-C command before a Docker or fixture operation. Recovery
     must compare every identity field before any Docker command and may not
     substitute a same-path binary, socket, endpoint, or daemon.
     `input_validated.detail` contains only
     `validation_scope="manifest_metadata"`, declared query bytes/SHA-256, and
     archive-manifest SHA-256. No fixture file has been opened; the task's source
     digest is the manifest-declared binding, not observed-content evidence;
  2. one CAS winner persists revision 2 `running` with `recall_started` before
     invoking `archive_controller.prepare()`. Prepare performs the Docker runtime
     preflight and then the one-descriptor secure reads from §4.3, but no transfer,
     index, rebuild, search, or candidate mapping;
  3. after prepare succeeds, CAS persists revision 3 `running` with
     `input_content_verified`. Its detail binds the complete canonical
     `verified_fixture_set_sha256`, entry/unique-source counts, query SHA-256,
     manifest digest, and runtime-preflight digest. Only then may
     `archive_controller.execute()` transfer the prepared immutable buffers and
     perform index/rebuild/search work;
  4. after execute and verified cleanup, the runner first writes and fsyncs every
     immutable non-summary success artifact plus a strict `success-intent.json`
     binding the proposed revision-4 task/pack bytes and pre-summary inventory.
     Only then may CAS persist revision 4 `evidence_ready` with
     `recall_completed`, `precision_started`, `precision_completed`, and
     `evidence_pack_ready`, plus the complete context and pack. It then closes the
     store, snapshots the inventory, and writes the summary last. A crash before
     CAS is recovered only to failed; a crash after CAS is completed only by the
     no-Docker finalizer below.
  No Docker call occurs before durable `recall_started`; no fixture file is opened
  before runtime preflight; and no drawing side effect occurs before durable
  `input_content_verified`. The seam never calls the legacy recall mapper or
  ordinary offline fallback and cannot turn a failed archive run into
  `insufficient_evidence`.
- `ReviewReuseTask.evidence_pack_schema_version` is a required-in-memory
  `Literal["evidence-pack-v1", "evidence-pack-v2"]` field. A persisted legacy
  record that lacks it loads as v1. Unknown values fail closed.
- `ReviewReuseTask.evidence_context` is a strict optional
  `ReviewReuseEvidenceContextV2` model. Its exact fields are frozen by the
  contract below. V1 requires it to be null. V2 requires null on the first
  three running snapshots because post-search facts do not yet exist, and requires the
  complete context for `evidence_ready`/`decided` snapshots before the pack can be
  persisted. A failed pre-pack ER3 task may retain null context but cannot expose
  a pack. Candidate receipt/search provenance remains in each strict candidate
  while the context carries the complete run-level digest bindings.
- `ReviewReuseTask.er3_run_binding` is a strict optional immutable model. It is
  null for v1 and required from revision 1 onward for v2, including failed tasks.
  Store create/put/CAS/load and recovery validate its exact fields and equality to
  the selector, manifest, idempotency preimage, runtime-preflight repository and
  Docker-control objects, final evidence context, and run summary. It is the
  ledger fact that lets recovery identify the exact run before `evidence_context`
  exists; no value may be inferred from a container label or output directory.
- The selector is set before the revision-1 pending/running task is first
  persisted and cannot change for the task lifetime. Store create, put, CAS,
  recovery, cancellation, decision reconstruction, and load validation all use
  the task-level selector; they never infer it from a missing or newer pack.
- Tests must prove event timestamps satisfy
  `recall_started <= prepare start <= prepare end <= input_content_verified <= execute start <= execute end <= recall_completed`
  and that a controller failure appends only the legal durable prefix plus
  `failed`; importing already completed external evidence after the fact is not
  an allowed chronology.
- Root creation, `control/` creation, private-Docker copy publication, control-
  receipt publication, store initialization, and revision-1 create happen under
  the external run-level lease. A crash before the complete revision-1 shape may
  leave only a strict prefix, which returns
  `archive_run_initialization_incomplete`; ordinary `er3-run` neither deletes nor
  repairs it. The only resumable revision-1 shape has exactly `control/docker`,
  `inputs/archive-manifest.json`, the six Python/repository/source/private/socket
  control receipts, the zero-byte store writer lease, one tenant sidecar, one task
  file, and an idempotency index if and only if the key is non-null. Their exact
  directory/file templates are frozen by
  `lifecycle_contract.pre_side_effect_inventories`. No temp/staging, setup,
  ownership, Docker-control, operation, success-intent, summary, ignored, linked,
  special, or extra path may exist. Every byte, mode, owner, digest, tenant/task
  identity, revision-1 event prefix, selector, idempotency preimage, and
  `ER3RunBinding` must match. Classification is ordered and mutually exclusive:
  a parsed revision-1-or-later task cannot be relabeled initialization-incomplete;
  exact revision-2 canceled and revision-3/4 failed roots have dedicated terminal
  branches, while only journal/receipt-prefix-valid revision-2/3 roots are
  recovery-required. Every other partial or extra shape is authorization mismatch.
  The CAS winner that appends `recall_started` is the sole controller owner. Any
  persisted v2 running task that contains
  `recall_started` but no terminal event is an interrupted run. Create retry,
  idempotency retry, restart, and ordinary `er3-run` must never call prepare or
  execute again for it.
- A v2 task at revision 2 or 3 cannot use the ordinary cancel transition. Cancel
  returns conflict `archive_run_recovery_required` without changing status,
  revision, events, binding, or controller ownership. Revision-1 v2 cancellation
  remains legal because no Docker action has occurred. The explicit recovery path
  below is the only transition out of an interrupted revision-2/3 task. Tests must
  cover cancel/controller/CAS/crash races and prove no terminal `canceled` task can
  retain journaled or run-id/resource-owner-id pair-labeled resources.
- Only explicit `er3-recover` may acquire the persistent-inode run-level lease and
  then the store writer lease, validate the exact tenant/task/run binding and the
  machine-readable recovery prefix grammar. Revision 2 admits an exact ordered
  prefix of Docker-environment, setup, resource-ownership, Docker-control,
  runtime-preflight, and verified-fixture-set artifacts; the verified fixture set
  is legal before the revision-3 CAS. Revision 3 admits, for operation `n=1..12`,
  only the global command prefix `inspection(2n-1), operation(n), inspection(2n)`
  plus prerequisite-valid
  index/search/service/ruleset, cleanup, runtime-seal, task/evidence, and success-
  intent prefixes; a complete success intent before revision-4 CAS is still
  recoverable only to failed. Each setup/operation/inspection/cleanup
  command is tested at `prepared`, `result_recorded`, every artifact-prefix,
  receipt-complete, journal-complete, and task-CAS crash boundary. Before cleanup,
  one legal journal-only successor must copy any normal pending command exactly
  into immutable `interrupted_command` and clear the active slot. Only a grammar-
  accepted root with no prior cleanup may select one recovery branch. Before
  container-create could have run and with no durable ID, recovery performs only
  three non-destructive pair-label container/volume/network lists; any match stops
  for manual handling and authorizes no delete. A durable container ID selects the
  four-command remove/container-list/volume-list/network-list branch. If create may
  have succeeded without a durable ID, exact zero-match discovery selects the
  three-command branch, while exactly one matching container may select the
  five-command discovery/remove/container-list/volume-list/network-list branch only
  when all three initial setup zero proofs are complete. Ambiguous discovery or a
  missing zero proof performs no deletion. Any cleanup command with a fully durable
  nonzero result atomically closes `cleanup_status="failed"` and issues no later
  command. An exact final zero-exit branch that still sees a labeled owned resource
  also closes failed without deleting that remaining resource. A grammar-accepted completed branch proves every
  container/volume/network resource absent and then CASes the interrupted task to
  `failed` with `code="archive_run_interrupted"`, `recovered=true`, and
  `cleanup_status="verified_absent"`. If process death occurs after that complete
  branch is durable but before task CAS, a later `er3-recover` may perform only the
  failed-task CAS and issues zero Docker commands. Recovery never opens fixture
  bytes, calls index/search, rebuilds evidence, or emits a success summary. If a crash occurred
  after an exact pre-revision-4 success-artifact prefix was fsynced, recovery keeps
  those immutable bytes as forensic artifacts but never exposes them as success,
  finalizes them, or permits replay-success from the failed task. A durable
  current-command result/artifact group with a nonzero exit, or a complete final
  branch with a non-empty owned-resource result, atomically records
  `cleanup_status="failed"` and returns `archive_cleanup_failed`; a prepared-only,
  partial, incomplete, uncertain, or malformed branch returns
  `archive_run_recovery_evidence_incomplete`.
  Neither state permits another automated Docker command. The only resumable
  post-cleanup shape is the exact complete verified-absent branch awaiting task CAS;
  every incomplete shape needs a separately owner-authorized manual procedure. An
  in-process failure likewise emits no success bundle and may persist `failed` only
  after verified cleanup.
- A persisted revision-4 task with an absent `run-summary.json` entry is
  `archive_run_finalize_required`, never an interrupted run. This is an ER3 CLI
  reason code, not a public API response. Only explicit `er3-finalize` may acquire
  the external run-level lease and then the store writer lease, verify the immutable binding,
  success intent, revision-4 store task, complete pre-summary artifact bytes,
  verified-absent owned resources, and absence of a conflicting summary. It enters
  a narrowly authorized store export-freeze that holds the store's in-process lock
  and writer lease, proves exactly one tenant sidecar, one named task, and the
  optional one-binding idempotency index and snapshots the full inventory. It then
  creates fixed `.run-summary.next` with exclusive/no-follow open, writes and file-
  fsyncs it, and uses a directory-FD-relative hard-link create as the no-replace
  publication step. It fsyncs the run-root directory, removes the candidate, fsyncs
  again, and requires one exact single-link summary before closing the store and
  releasing the run-level lease. It performs
  no Docker, fixture, index/search, task transition, pack reconstruction, or
  external call. A preexisting exact externally hash-bound summary is already
  complete; any other preexisting summary entry is a hard mismatch and is never
  overwritten, unlinked, or repaired. A mismatch leaves revision 4 unchanged and returns
  `archive_completion_incomplete`; it never creates a second output root or
  re-executes side effects. Crash tests cover root creation, revision 1, every
  revision-4-finalization boundary, and idempotent re-entry after summary write.
- Existing and default API tasks remain v1 and reproduce the exact current raw
  hit mapping, canonical pack bytes, and digest. Only the internal ER3 runner
  explicitly requests v2. API task responses use a dedicated required-field
  `ReviewReuseTaskResponse`; EvidencePack responses use a v1/v2 discriminated
  union keyed by `schema_version`. Create/get/cancel/decision and list summary all
  expose the required selector, and a non-null pack must match it. The public
  create request exposes no selector or media override. This requires an updated
  OpenAPI snapshot plus property-level tests; decision request fields do not
  change.
- A v1 keyed create retains the exact current digest preimage:
  `{"tenant_id": <tenant>, "source_content_sha256": <sha256>}`.
- A v2 keyed create uses exactly `tenant_id`, declared source-content SHA-256,
  `evidence_pack_schema_version="evidence-pack-v2"`, and the complete normalized
  `ER3RunBinding`. Reusing a key across v1/v2 or across any changed run-binding
  field conflicts. Unkeyed v2 creates still persist the binding while retaining
  null idempotency key/digest metadata.
- Decision idempotency preimages do not add a redundant selector because they
  already bind `evidence_pack_sha256`; decision reconstruction retains the task
  selector after temporarily removing the reviewed pack.
- The v1 builder and mapper remain frozen. A new v2-only recall mapper and pack
  builder are selected from the immutable task field; shared v1 score coercion is
  not corrected in place.
- V1 keeps its historical strategy-center `allowed_actions` array byte-for-byte.
  V2 instead names the complete five-state schema as `decision_vocabulary` and
  never represents it as runtime availability. The ER3 runner leaves
  `human_decision.state/submitted` null and decisions disabled; only a later
  owner action can make submission available. A non-null v2 submitted decision
  requires `reviewer_kind="validated_principal"` and existing reason-code/CAS
  validation.
- Reading a legacy v1 record never rewrites it. A later authorized task write may
  materialize the default v1 selector while preserving the existing pack digest
  and normal revision/event semantics; there is no bulk migration.
- Old v1 running, evidence-ready, canceled, and decided fixtures must load and
  retain their exact idempotency/pack/decision digests before and after this
  tranche. No existing pack is silently upgraded.
- Calibration validation is selector-conditioned. V1 continues to require its
  current non-null `calibration_version`; a null v1 value is corrupt even if a
  rebuilt v1 pack would also be null. For the approved uncalibrated v2 run,
  version and digest are both null with an explicit reason. A calibrated v2 pack
  requires both non-null. Selector, context, pack schema, calibration, or
  provenance mismatch fails closed on create, load, CAS, cancellation, decision
  reconstruction, and replay.

The complete v2 canonical contract is
`docs/development/L3_REVIEW_REUSE_ER3_EVIDENCE_PACK_V2_CONTRACT_20260829.json`
with contract digest
`680c596d315061424a32d160d95131b7da9acc1a02a238397b2f83c40da8a372`.
Its reviewed raw-file SHA-256 is
`9f30ee2c2f4e57180538ce493d28e21c7b8f8e75061cbafeabfad1a5153e1e96`.
It freezes exact object keys, types/nullability, list ordering, task context,
unknown-field rejection, and digest exclusions. Its embedded golden vector has
EvidencePack digest
`422e23da3589b24a5539a3d6546cac98ba692046ea14930b179dd9f7fe1b9f7f`.
The contract digest excludes only `contract_sha256`; the golden EvidencePack
digest excludes only `evidence_pack_sha256`. Implementation copies neither value
from prose: tests recompute both with `canonical_json_v1()` and reject any field,
type, ordering, nullability, or digest drift.

The contract carries `contract_status="proposed_not_ratified"`,
`runtime_authority="none"`, and `proposed_run_bindings`. Those bindings become
runtime authority only across the explicit §10 Gate-A, Gate-B, and Gate-C owner
responses that name the exact design-lock, fail-first, implementation, and runtime
heads. No caller-supplied digest or contract field can self-ratify this FOR REVIEW
revision.

This is a narrowly scoped L3 compatibility change. Gate A cannot authorize
runtime code, and Gate B cannot start until its exact fail-first predecessor is
accepted. Even after Gate B, image runtime remains blocked until Gate C in §10.

### 5.5 Provenance

The strict persisted `ReviewReuseEvidenceContextV2`, candidate provenance, and
success bundle together bind actual observed values or canonical digests for:

- source query SHA-256;
- archive manifest SHA-256;
- successful canonical index receipt and receipt-set digests;
- post-index search-response receipt digest and manifest archive-entry identity;
- service identity/version payload digest;
- model identifier/version/digest when the service exposes it;
- ReviewReuse score-mapping ruleset digest;
- calibration status and version/digest, with null for unavailable fields;
- runtime preflight and continuous runtime-seal receipt digests;
- exact clean owner-authorized repository commit, reviewed-command digest, and
  runner version, transitively bound by the runtime-preflight receipt;
- task, trace, idempotency, task revision, and EvidencePack identifiers.

The v2 contract now freezes the complete canonical preimage, exact fields, nested
field schemas, invariants, and non-placeholder golden for each critical digest:

- search response receipt:
  `b4ff738ada27842e09ff22791b52cc504860f589d23e6cbcc18bc9e542975602`;
- service identity:
  `b9defaaa4689ea63652663cc5431c8c431a23a89d31f4f0227908ee49278660e`;
- score mapping ruleset:
  `ccee15b504054a1cd3def3f6531babbc41a72742d3d2dbb8e35efd57337afd11`;
- Docker control plane:
  `ef6802d6138b19db4f580a75ac83fdf99b6d04e2bd3c56199d4e0484ca6c5ea7`;
- strict runtime inspect projection:
  `c907c2aa218065321eba9ac471a4436286f363632809fcbd8c15a8c6838cf927`;
- runtime preflight:
  `bc22e7cbdecaf7be460a10165e03f60df238dd1741e2353728dbd4379daf5e3c`;
- continuous runtime seal:
  `8d9da40c31be1770c4a38f44c4e6303b34ad689346f6496ba91266ba1b5297f0`.

Those values are serialization vectors, not expected live observations. A live
run exports and hashes its own complete named preimages. All-zero, one-character
repeated, missing-preimage, non-recomputed, or copied-golden-without-an-identical-
observed-preimage digest values fail.
If nullable model/calibration metadata is unavailable, all fields in that branch
are null with one structured reason; partial metadata combinations are invalid.
A run that cannot establish archive/index/search/runtime provenance cannot close
ER3. Every field needed to rebuild the v2 EvidencePack after process restart lives
in the task or its candidates; exported receipts are evidence bound by digest,
not an unstated reconstruction dependency.

### 5.6 Export and replay

`scripts/review_reuse_isolated_archive_run.py` owns four explicit ER3 subcommands:
`er3-run`, `er3-replay`, `er3-recover`, and no-Docker `er3-finalize`. The current no-subcommand invocation retains its v1
argument/control-flow compatibility for the existing Makefile target and
regression tests, and its v1 EvidencePack bytes/digest remain frozen. Whole task
JSON and audit exports may add the required explicit v1 selector; they are not
claimed byte-identical. The bare path is never an ER3 mode or success fallback.
Any argv beginning with an ER3 subcommand is parsed exclusively as that
mode and fails closed instead of falling through to legacy behavior. Replay support is
implemented in that already locked script plus the named ER3 modules from §8; it
may not introduce an unnamed module.

A success run writes the filesystem store, manifest copy, verified-fixture-set,
repository/socket/binary identities, runtime preflight, continuous runtime-seal
receipt, resource ownership, success intent, and every receipt preimage required
to recompute it. For each of the nine
setup, 12 sealed-operation, and four ordinary-success cleanup Docker commands, the run contains one
canonical command-receipt JSON plus exact raw stdin, stdout, and stderr byte artifacts.
For each of the 24 fresh inspections, it additionally contains the canonical
inspect-receipt JSON, its command-receipt JSON, and that command's exact raw stdout
and stderr plus its zero-byte stdin; equal stream bytes do not permit a path or
observation to be omitted.
The run also writes strict raw index responses, normalized receipts and receipt
set, repository-control receipt, redacted rebuild/search/service receipt preimages,
task JSON, EvidencePack JSON, EvidencePack Markdown, audit bundle, store writer
lease, and run summary.
The separate run-level lease remains outside the run root and is not an artifact.
The v2 contract freezes every required non-store artifact path template, count,
encoding, and digest relationship; only files beneath the explicit filesystem
store use one closed tenant/task/idempotency family. `control/docker`, the exact
Docker environment preimage, and the final run journal are mandatory long-lived
artifacts. Before inventory, the runner enters the store export-freeze and proves
its zero-byte writer lease remains held while retaining the external run-level
  lease through summary publication and parent fsync. Both locked FDs must still match
their permanent path device/inode; neither lease path may be unlinked, renamed,
truncated, or replaced. `run-summary.json` is
canonical JSON with exactly these top-level fields:
`schema_version`, `status`, `run_id`, `tenant_id`, `task_id`, `task_revision`,
`evidence_pack_sha256`, `archive_manifest_sha256`,
`runtime_preflight_sha256`, `runtime_seal_receipt_sha256`,
`success_intent_sha256`,
`repository_commit`, `reviewed_command_sha256`, `runner_version`, `store_root`,
`writer_lease`, and `artifacts`. For success, schema is
`review-reuse-er3-run-summary-v1`, status is `succeeded`, and task revision is 4.
`writer_lease` has exactly `relative_path`, `sha256`, `size_bytes`, and
`held_through_summary_fsync`; `artifacts` is the lexicographically keyed map from
every other regular file's ASCII POSIX relative path to an exact
`{sha256,size_bytes}` object. Missing any required template instance, duplicate
sequence/ordinal, or adding a non-store file outside the closed template set fails.
The summary itself is the only excluded final file and is written last through the
fixed `.run-summary.next` no-replace protocol. Neither summary/journal candidate nor
any other temporary path may survive closed inventory. The success stdout records
its SHA-256; replay requires that value
as `--summary-sha256` rather than trusting the file that names its own children.

Artifact paths forbid absolute paths, empty segments, `.`, `..`, backslashes,
non-ASCII bytes, and duplicate normalized results. Recursive `lstat` accepts only
directories and regular files with `st_nlink == 1`; symlink, hardlink, socket,
FIFO, device, path escape, missing file, or extra file fails. The store root and
store writer lease are relative paths below the run root, and that lease is
included in the artifact map. Its open FD and path `lstat` identity,
owner, mode, link count, and zero size must remain equal from store-lock acquisition
through summary parent fsync. The store subtree must contain exactly
one hashed tenant directory, one sidecar, one task, and only the conditional
single-binding idempotency index; another tenant/task/index or any staging/temp
entry fails. The complete live directory set is also exact: it includes
`control`, `inputs`, `evidence`, `task`, `state`, the enumerated receipt directories,
and the closed dynamic store family. The mode-`0700` `docker-config-empty` and
`docker-home-empty` directories are required and recursively empty before and after
every Docker command and at final inventory; another empty or unlisted directory
fails. The external run-level lease is deliberately excluded but its stable inode
remains locked through summary fsync.

The v2 Markdown bytes are frozen as:

```python
b"# ReviewReuse EvidencePack v2\n\n" \
b"Canonical evidence (null remains unavailable):\n\n    " \
+ canonical_json_v1(validated_persisted_pack) \
+ b"\n"
```

Pretty JSON, `repr`, locale, wall-clock time, or host paths are forbidden.
`evidence.md` equals those bytes; the audit bundle Markdown is their strict UTF-8
decode. The golden bytes are 3,556 bytes with SHA-256
`6e5778233d90887932e2eea2a182df9563f4af03804467098427b6e00b9eeac2`.
The minimal run-summary serialization vector has digest
`8998984f0745ec0c3177597e8e1258ccdc45588742b06eae277ffb8d8f2d6d0f`.
V1 rendering remains byte-identical.

Replay first acquires a shared lock on the same permanent run-level lease used
exclusively by run/recover/finalize, then has one ordered fact chain:

1. open the already-existing permanent run-level lease read-only without
   `O_CREAT`, then strictly parse the summary and match the required externally
   supplied summary SHA-256;
2. walk by directory FD, `fstat` and hash the complete run-root inventory before
   opening the store and
   reject missing, extra, renamed, linked, special, or mismatched artifacts;
3. recompute every command receipt from its canonical JSON and raw stdin/stdout/
   stderr, validate every direct multipart operation's exact `/usr/bin/curl
   --disable` argv, manifest-bound filename/type, and raw stdin, and
   recompute every inspect receipt and strict raw HostConfig plus raw `Config.Env`
   projection from its recorded inspect stdout, every cleanup receipt,
   verified-fixture-set, resource ownership, success intent, preflight, seal,
   index receipt/set, search, service, ruleset, and their cross-bindings without
   invoking Docker;
4. open the run filesystem store in explicit read-only mode and load the exact
   `(tenant_id, task_id)` through ordinary selector-dispatched store validation;
5. require canonical equality among the loaded task, exported task JSON, the
   task-embedded pack, and independent EvidencePack JSON;
6. validate the pack and contract/golden rules, then regenerate only Markdown and
   audit rendering and require byte equality with their recorded artifacts;
7. repeat the complete run-root inventory after replay and require no write,
   repair, lease/sidecar creation, event, revision, or file change.

Replay never rebuilds the canonical pack JSON, calls Docker/index/search, or uses
an exported EvidencePack as the primary fact source. Each artifact class and the
summary hash have independent tamper tests. Markdown and audit regeneration is
in-memory; `er3-replay` writes no file, event, revision, or store metadata. Any
mismatch fails before emitting a success report. Control receipts intentionally
retain originating-host absolute interpreter/Git/Docker/socket paths; this bundle
is therefore replay-verifiable on that host, not a portable export. A path-rewriting
portable bundle would require a separate design lock.

`er3-recover` is not replay: it is an explicit mutating failure-only command with
the narrow interrupted-run behavior in §5.4. It never emits or repairs a success
bundle. `er3-finalize` is also not replay: it may publish only the already-fsynced,
intent-bound revision-4 bundle and final summary, with no task or Docker effect.

## 6. Failure taxonomy

The CLI exits non-zero with a structured `status="failed"` and one stable
`reason_code` from this minimum vocabulary:

- `manifest_invalid`
- `manifest_content_drift`
- `archive_input_media_invalid`
- `archive_control_transport_invalid`
- `archive_docker_endpoint_invalid`
- `archive_host_environment_invalid`
- `archive_run_authorization_mismatch`
- `archive_substrate_unattested`
- `archive_runtime_preflight_failed`
- `archive_runtime_drift`
- `archive_instance_not_isolated`
- `archive_index_cardinality_unavailable`
- `archive_index_readiness_failed`
- `archive_search_cache_unsafe`
- `archive_recall_incomplete`
- `archive_run_interrupted`
- `archive_run_initialization_incomplete`
- `archive_run_terminal_canceled`
- `archive_run_terminal_failed`
- `archive_run_recovery_evidence_incomplete`
- `archive_run_writer_conflict`
- `archive_run_recovery_required`
- `archive_run_finalize_required`
- `archive_completion_incomplete`
- `archive_resource_ownership_conflict`
- `vision_health_unavailable`
- `archive_index_failed`
- `archive_index_request_invalid`
- `archive_rebuild_failed`
- `archive_search_failed`
- `archive_provenance_unavailable`
- `score_source_invalid`
- `evidence_contract_invalid`
- `export_replay_mismatch`
- `archive_cleanup_failed`

A failure artifact may contain paths relative to the approved fixture root,
digests, reason codes, and redacted error classes. It must not contain secrets,
raw private drawings, URL credentials, or unrestricted upstream response text.

## 7. Required fail-first tranche

After owner ratification and before ER3 implementation, add
`tests/unit/test_review_reuse_er3_archive.py` with these exact tests and attach a
classified baseline log against the ratified runtime base as defined below:

1. `test_er3_cli_bootstraps_without_pythonpath`
2. `test_er3_exact_manifest_digest_and_file_metadata_are_pinned`
3. `test_er3_manifest_uses_png_media_and_rejects_direct_dxf`
4. `test_er3_manifest_rejects_path_escape_hash_drift_and_duplicate_roles`
5. `test_er3_query_is_not_added_to_archive_index`
6. `test_er3_real_mode_rejects_seed_candidates`
7. `test_er3_real_transport_has_no_host_endpoint_or_socket`
8. `test_er3_docker_exec_uses_fixed_argv_noproxy_and_no_shell`
9. `test_er3_requires_attested_index_roots_and_cardinality_contract`
10. `test_er3_static_attestation_cannot_replace_runtime_preflight`
11. `test_er3_requires_fresh_digest_pinned_ephemeral_vision_instance`
12. `test_er3_container_uses_network_none_without_published_ports`
13. `test_er3_numeric_bounded_egress_probe_requires_curl_exit_7_and_network_none`
14. `test_er3_index_count_is_zero_then_matches_archive_entry_count`
15. `test_er3_fresh_instance_has_zero_preindex_query_results`
16. `test_er3_preindex_probe_cannot_poison_postindex_search_cache`
17. `test_er3_drawing_commands_execute_exactly_once`
18. `test_er3_index_add_disables_object_store_upload`
19. `test_er3_health_index_and_search_fail_closed`
20. `test_er3_rebuild_populates_both_index_layers`
21. `test_er3_postindex_search_requires_receipt_bound_exact_candidate`
22. `test_er3_success_true_with_empty_or_unknown_candidates_fails`
23. `test_er3_visual_score_is_not_copied_to_semantic_or_geometric`
24. `test_er3_missing_geometry_is_null_and_marks_vision_only_unverified`
25. `test_er3_uncalibrated_run_does_not_emit_confidence`
26. `test_er3_provenance_rejects_placeholders_and_binds_observed_digests`
27. `test_er3_internal_v2_accepts_only_manifest_bound_png`
28. `test_er3_public_v1_api_still_rejects_same_png_with_415`
29. `test_er3_v2_never_disguises_png_as_dxf`
30. `test_er3_v2_contract_and_golden_digests_are_pinned`
31. `test_er3_v2_envelope_forbids_unknown_fields_and_type_drift`
32. `test_er3_v2_persisted_context_rebuilds_identical_pack_after_restart`
33. `test_er3_task_selector_is_immutable_and_legacy_defaults_to_v1`
34. `test_er3_v1_raw_mapping_pack_and_decision_digests_are_unchanged`
35. `test_er3_v1_and_v2_create_digest_preimages_are_exact`
36. `test_er3_store_dispatches_v1_and_v2_without_silent_upgrade`
37. `test_er3_cancel_and_decision_reconstruction_retain_task_selector`
38. `test_er3_unknown_evidence_pack_version_fails_closed`
39. `test_er3_v1_rejects_null_calibration_while_v2_uncalibrated_requires_null`
40. `test_er3_selector_context_pack_and_calibration_tampering_fail_closed`
41. `test_er3_openapi_requires_selector_and_matching_pack_without_create_switch`
42. `test_er3_replay_reemits_persisted_pack_without_network_calls`
43. `test_er3_replay_verifies_store_task_pack_summary_and_artifact_hashes`
44. `test_er3_replay_rejects_each_tampered_artifact_without_network`
45. `test_er3_json_markdown_and_audit_export_are_consistent`
46. `test_er3_runner_never_enables_or_submits_decisions`
47. `test_er3_dedicated_instance_and_volumes_are_cleaned_up`
48. `test_er3_docker_daemon_is_the_only_real_run_control_boundary`
49. `test_er3_index_receipt_and_set_canonical_vectors_are_pinned`
50. `test_er3_candidate_and_context_receipt_digests_match_canonical_receipts`
51. `test_er3_index_add_requires_fixed_user_name_and_upload_query`
52. `test_er3_rejects_remote_or_environment_overridden_docker_endpoint`
53. `test_er3_host_config_forbids_privilege_caps_devices_namespaces_and_binds`
54. `test_er3_private_shm_and_all_mutable_paths_are_isolated`
55. `test_er3_rejects_ambient_store_live_provider_proxy_and_existing_output`
56. `test_er3_query_direct_upload_search_happens_exactly_twice`
57. `test_er3_critical_provenance_preimages_and_golden_digests_are_pinned`
58. `test_er3_visual_only_ruleset_does_not_copy_supplier_duplicate_verdict`
59. `test_er3_live_success_rejects_zero_repeated_or_unbound_digest_placeholders`
60. `test_er3_archive_controller_execution_is_bracketed_by_recall_events`
61. `test_er3_v2_running_context_is_null_then_required_for_evidence_ready`
62. `test_er3_runtime_posture_drift_between_control_operations_aborts`
63. `test_er3_replay_is_an_explicit_read_only_script_subcommand`
64. `test_er3_v2_decision_vocabulary_does_not_claim_runtime_availability`
65. `test_er3_v2_reviewer_kind_and_rejection_vocabularies_are_closed`
66. `test_er3_required_ci_gate_runs_exact_fail_first_contract`
67. `test_er3_run_requires_clean_exact_authorized_head_and_reviewed_command`
68. `test_er3_runtime_preflight_binds_static_attestation_digest`
69. `test_er3_inspect_projection_binds_image_command_user_lifecycle_and_network_state`
70. `test_er3_fixture_bytes_are_loaded_only_after_preflight_and_before_transfer`
71. `test_er3_transfer_content_hashes_bind_the_exact_manifest_bytes`
72. `test_er3_legacy_cli_remains_v1_only_and_er3_subcommands_never_fallback`
73. `test_er3_cleanup_failure_emits_no_success_bundle`
74. `test_er3_revision_one_input_validated_is_manifest_metadata_only`
75. `test_er3_fixture_content_verified_is_durable_after_preflight_before_transfer`
76. `test_er3_recall_started_is_durable_before_archive_controller`
77. `test_er3_interrupted_running_task_never_reinvokes_archive_controller`
78. `test_er3_interrupted_run_recovery_cleans_then_fails_without_success_bundle`
79. `test_er3_run_summary_schema_and_complete_artifact_inventory_are_exact`
80. `test_er3_run_summary_rejects_path_escape_symlink_hardlink_and_extra_artifact`
81. `test_er3_run_lease_blocks_finalize_race_while_store_lease_is_inventory_bound`
82. `test_er3_v2_markdown_bytes_are_deterministic_and_audit_identical`
83. `test_er3_implementation_ancestry_descends_from_updated_er1_er2_and_design_lock_heads`
84. `test_er3_control_receipts_bind_full_argv_endpoint_socket_api_binary_and_streams`
85. `test_er3_inspect_receipts_are_fresh_one_to_one_and_not_cached`
86. `test_er3_docker_client_environment_and_binary_quarantine_fail_closed`
87. `test_er3_closed_host_config_rejects_unlisted_or_relaxed_controls`
88. `test_er3_fixture_paths_runtime_names_and_reads_are_nofollow_regular_and_bounded`
89. `test_er3_out_rejects_symlink_components_and_enforces_new_mode_0700_root`
90. `test_er3_revision_one_persists_run_binding_and_v2_create_digest_binds_it`
91. `test_er3_active_v2_cancel_conflicts_and_cannot_escape_recovery`
92. `test_er3_owner_bound_socket_identity_api_version_and_empty_docker_config_are_exact`
93. `test_er3_raw_hostconfig_v143_mapping_rejects_unknown_missing_and_neutral_drift`
94. `test_er3_fixture_open_uses_directory_fd_chain_and_rejects_ancestor_swap`
95. `test_er3_verified_fixture_set_binds_every_manifest_entry_before_execute`
96. `test_er3_replay_recomputes_every_command_stream_inspect_and_cleanup_receipt`
97. `test_er3_gate_a_collects_classified_nodes_without_src_or_runner_changes`
98. `test_er3_python_bootstrap_rejects_shadow_src_pythonhome_user_site_and_pth`
99. `test_er3_private_docker_copy_blocks_source_rename_between_check_and_spawn`
100. `test_er3_direct_multipart_stdin_rejects_hash_size_trailing_and_filename_drift`
101. `test_er3_raw_config_env_rejects_duplicate_unknown_missing_order_and_value_drift`
102. `test_er3_external_json_rejects_underflow_overflow_and_lossy_binary64_numbers`
103. `test_er3_revision_one_binds_python_binary_socket_parent_and_owner_id_for_recovery`
104. `test_er3_existing_root_resumes_only_revision_one_and_finalizes_only_revision_four`
105. `test_er3_event_enum_and_api_conflict_map_cover_verified_finalize_and_recovery`
106. `test_er3_run_id_zero_label_proof_and_resource_journal_are_exclusive`

Tests may use a deterministic fake private vision server and pure mocked Docker
CLI/subprocess/inspect responses for failure, mapping, and isolation-posture cases.
The first gate does not authorize a Docker-daemon connection, image inspection,
pull, start, or fixture processing. ER3 closure additionally requires a separately owner-authorized recorded
run against the dedicated pinned `dedupcad-vision` container and the approved
manifest. Mock-only green tests are not closure evidence.

The runtime-base and candidate-Gate-A-head logs are governed by
`lifecycle_contract.implementation_ancestry.gate_a_fail_first_matrix`, not by a
Gate-A author's judgment. Against exact base `af72d0ec02b5...`, tests 28 and 34
are the only `expected-existing-pass` nodes; the other 104 are `expected-red` at
their exact `missing_behavior::<test-id-suffix>` boundary. The contract contains
all 106 `test_id -> expected_baseline -> reason_code -> required_failure_boundary`
entries and the 104/2 base summary. At the candidate Gate-A head, sequences 2, 28, 30, 34,
66, and 97 are the exact six expected passes because the manifest/contract,
existing v1 guards, required workflow, and classifier self-check then exist; the
other 100 remain red. `pytest --collect-only` must enumerate the same exact set in
both phases, and both aggregates exit nonzero. Every expected-red node must reach
its named boundary and call `pytest.fail(required_failure_boundary, pytrace=False)`;
the JSON report must show setup/teardown pass, call fail, and exact longrepr
`Failed: <required_failure_boundary>`. Collection/import/setup/teardown failure,
generic assertion/exception, wrong marker, deselection, skip, xfail, xpass,
unexpected pass/fail, duplicate, or unclassified node is invalid evidence.

The verifier observes the candidate Gate-A checkout's exact commit; this is an
evidence subject, not owner acceptance. It resolves runtime base, ratified design-
lock head, and candidate head as raw commit objects and proves
`runtime-base -> design-lock -> candidate-Gate-A`. Every Git command uses one
reviewed absolute binary, `--no-pager`, `--no-optional-locks`,
`--no-replace-objects`, `--no-ext-diff` where accepted, a closed five-key
environment, and fixed helper-disabling `-c` values. System/global config is
disabled. Repository-local config is independently enumerated with origin/scope,
must contain only the contract's inert `core`/`remote`/`branch` allowlist, and is
bound into a strict repository-identity receipt; include/includeIf, worktree
config, aliases, filters, helpers, hooks, pagers, alternates, and every other key
fail. Replacement refs, grafts, and alternates must be absent and the repository
must be non-shallow. The verifier independently recomputes every consumed
non-quarantined SHA-1 commit/tree/blob ID from its raw object header/body, walks raw
parent links for ancestry, and walks raw trees for deltas and fixture-elided
materialization; `merge-base` and `diff` are corroborating receipts only.

It materializes a fixture-elided projection of the runtime-base raw tree into an
empty execution root, rejecting pre-existing bytes, sparse/filter output,
alternate objects, and any skip-worktree or assume-unchanged flag. The verifier
walks only tree entries below `tests/vision/fixtures/cad_features`, records the
directory tree ID and the three child path/mode/blob IDs, and never requests those
blob bodies, types, sizes, headers, or raw hashes. Every other materialized regular
file retains path/mode/blob/size/raw-SHA-256 evidence. From the exact candidate
Gate-A object, it overlays only the blob at
`tests/unit/test_review_reuse_er3_archive.py`; no contract, fixture, verifier,
workflow, helper, runtime, or source path may be copied into that execution root.
The verifier executes from a separate read-only export of its exact candidate-head
Git blob, and reads the ratified contract from the exact design-lock Git object
outside the execution root.

Both pytest phases use an owner-reviewed absolute Python 3.11 interpreter under
`-I -S` plus a stdlib-only bootstrap. Interpreter identity, sealed dependency
manifest, exact argv/environment, final `sys.path`, and every source/test/dependency
import mapping are digest-bound; user site, `.pth`, editable/zip installs,
`PYTHONPATH`, `PYTHONHOME`, CWD imports, and unknown roots fail. The materialized
source is read-only. The materializer creates only an empty trusted mountpoint at
the quarantined fixture path; a root-owned mode-`000` empty directory is then
deny-mounted there. As the test UID, fixed negative `openat` probes for the three
PNG names must return `EACCES`, and the test sees neither pre-opened fixture FDs
nor Git objects. Fixture payload bytes are never opened, statted, inflated, sized,
hashed, or materialized by either privileged setup or the test.

Pytest runs as a distinct unprivileged UID. One private mode-`0700` tmpfs scratch
root outside source is its only writable location and supplies `TMPDIR` and
`--basetemp` for synthetic filesystem tests. A privileged supervisor owns the
separate report root and captures pytest events; the test cannot write that root.
Mount, fixture-deny, negative-open, scratch, interpreter/import, supervisor-capture,
and pre/post materialization receipts are all bound into each execution object. A
second empty fixture-elided root is materialized from the candidate Gate-A tree and
runs without overlay under the same policy. Any Git-view, object, import,
read-deny, scratch,
mount, pre/post, overlay, receipt-preimage, or report-capture mismatch is invalid
evidence. Every execution digest resolves through the contract's exact retained
path, raw-versus-canonical encoding, and closed receipt schema; a generic
"hash the named artifact" convention is not accepted.

The only classifier implementation is
`tests/verification/review_reuse_er3_contract_verifier.py`. Gate A runs its exact
candidate-head Git blob outside both execution roots. The privileged verifier
receives the exact ratified contract object and runtime-base SHA, launches both
pytest phases itself, and captures their event evidence into a test-unwritable
report root; it does not trust a test-authored JSON report. It emits one strict
`review-reuse-er3-gate-a-verification-v1` JSON object outside the candidate commit,
with recursively closed fields, types/nullability, enums, ordering, and every
sub-digest preimage frozen in the machine contract. It binds a strict producer CI
identity sourced only from the protected GitHub Actions context and immutable event
payload: repository name/ID, pull-request head, workflow path, run/attempt, fixed
`er3-contract` job, and artifact name. The raw-object candidate head must match that
producer head, and no workflow input or caller override is accepted. It contains a candidate
artifact-set object for the exact eight singleton design/runbook/workflow/test/
manifest/attestation/verifier artifacts plus every additional strict-JSON fixture,
and binds that object with its own self-excluding canonical digest. It also contains
the 106/104/2 runtime-base and 106/100/6 Gate-A-head counts plus exactly 212 phase-
qualified classified entries, each retaining the matrix `reason_code`, failure
boundary, observed pytest phases, and exact longrepr. The report has `valid=true`
only when both
pytest aggregates remain nonzero while every expected-red/pass, object, ancestry,
raw-object/import/read-deny/materialization and digest proof succeeds. A zero pytest exit in
either phase invalidates Gate A. This report and all retained receipts are emitted
as one immutable CI artifact only after the candidate commit exists; they are not
written into that commit and cannot self-ratify. A later owner response must name
the candidate head, workflow run/job/artifact identities, archive digest, report
raw/full-canonical/self-excluding verification digests, and the exact candidate
artifact-set digest before it becomes accepted Gate-A evidence. The classifier
namespace `reason_code` is only a
Gate-A expectation label and is distinct from the runtime CLI failure vocabulary
in §6. The verifier must hardcode the owner-ratified contract canonical/raw digests;
reading expected values solely from the contract's own self-digest is forbidden.

Within the fixed 106 node IDs, tests 67, 79-81, 84, 86, 97, 104, and 106 are
parameterized to cover respectively out-of-band Gate-C integrity, closed store/
control/environment/journal inventory, persistent lease inode, real argv plus
per-command identities, the exact environment artifact, fail-first classification,
mutually exclusive terminal/root states, and every journaled crash boundary. No
extra unclassified node substitutes for these required parameter sets.

At least one mutation case must make each Python import, locality, private Docker
binary/endpoint/argv/stdin, fresh-inspection, raw HostConfig/Config.Env, fixture-
descriptor/direct-stdin, fixed-query, continuous-seal, chronology/recovery/finalization,
resource-ownership, artifact-inventory, lossless-number, canonical-preimage,
Markdown, ruleset, and replay guard fail;
asserting only happy-path output is insufficient. Renaming, weakening, deleting,
skipping, or marking these tests xfail requires
owner review. Additional narrower tests are allowed.

## 8. Proposed gated write sets

### 8.1 Gate A: fail-first artifacts only

The first owner response may authorize only these new files:

- `tests/unit/test_review_reuse_er3_archive.py` containing exactly the 106 names in
  §7 and using in-test dynamic imports so missing runtime modules produce ordinary
  assertion failures rather than collection/import errors;
- `tests/fixtures/review_reuse_er3/archive_manifest.json`, byte-equivalent to the
  embedded proposal in §4 after canonical serialization;
- one or more synthetic Docker-inspect/API-v1.43 and receipt-stream fixtures below
  that same fixture directory. Every additional fixture is a mode-`100644` strict
  I-JSON `.json` data blob with a closed basename; Python, `conftest.py`, executable,
  configuration/autoload, symlink, and submodule entries are forbidden;
- `.github/workflows/review-reuse-er3-contract.yml` with the contract below;
- `tests/verification/review_reuse_er3_contract_verifier.py`, the sole baseline
  classifier and canonical/digest verifier named above;
- `docs/development/L3_REVIEW_REUSE_ER3_GATE_A_FAIL_FIRST_VERIFICATION_20260830.md`,
  a pre-execution runbook/template that contains no same-head run ID or report
  digest. The strict JSON result is an out-of-commit CI artifact, not this file.

Gate A forbids changes to `src/`, `scripts/review_reuse_isolated_archive_run.py`,
existing tests/fixtures, deployment, configuration, and runtime code. The Gate-A
candidate exact head must descend from the ratified design-lock head and retain an aggregate
nonzero test result with every one of the 106 nodes classified as expected-red or
expected-existing-pass. It is evidence for a later owner decision, not a mergeable
or production-ready head.

The Gate-A workflow must implement the exact raw-object/external-verifier/single-test-file
overlay protocol in §7 and the machine contract. Its runtime-base checkout may
contain no Gate-A file other than that one test blob, while the Gate-A-head phase
uses a separate clean checkout. Any shortcut that runs both reports from the
Gate-A worktree or trusts a claimed SHA instead of observed Git state fails the
gate. The same workflow and verifier must contain their Gate-B verification mode
at Gate A; both become immutable inputs after the owner accepts that exact head.
The workflow uploads the strict report plus all digest-preimage receipts as the
fixed immutable artifact `review-reuse-er3-gate-a-evidence`. Neither branch content
nor PR metadata supplies owner authority; they produce candidate evidence only.
The verifier also records the exact design-lock-to-Gate-A Git diff tree. Every
entry must be a new mode-`100644` regular blob at one of the five Gate-A-added singleton
paths above or an additional synthetic fixture strictly below
`tests/fixtures/review_reuse_er3/`; all five singletons and at least one additional
strict-JSON synthetic fixture are required. Modification, deletion, rename, copy,
wrong mode/type, code/autoload fixture, symlink, submodule, or any other path
invalidates Gate A.

### 8.2 Gate B: mock-backed runtime implementation

Only after the owner separately names and accepts the exact Gate-A fail-first head,
Gate B may authorize:

- `scripts/review_reuse_isolated_archive_run.py` only for explicit `er3-run`,
  read-only `er3-replay`, failure-only `er3-recover`, and no-Docker
  `er3-finalize` subcommands; the existing bare/no-subcommand
  synthetic v1 behavior retains argument/control-flow compatibility and frozen
  v1 EvidencePack bytes/digest for current Makefile/tests, is never an ER3 mode,
  and can never receive fallback from an ER3 subcommand; whole task/audit exports
  may add the explicit v1 selector and are not claimed byte-identical
- new `src/core/review_reuse/er3_archive.py`
- new `src/core/review_reuse/evidence_v2.py`
- new `src/core/review_reuse/evidence_dispatch.py`
- `src/core/review_reuse/service.py` and `store.py` only for the explicit v1/v2
  selector, internal manifest-bound PNG seam, read-only replay, dispatch,
  context validation, idempotency rules, persistent run/store lease-inode
  validation, and the narrow export-freeze that keeps the writer lease held
  through summary fsync in §5.4-§5.6
- `src/core/review_reuse/models.py` only for the immutable selector,
  `ER3RunBinding`, `input_content_verified` event, strict v2 evidence context, and
  selector-conditioned calibration described in §5.3-§5.4
- `src/api/v1/review_reuse.py` only to add strict task/EvidencePack response
  models, the selector to `TaskSummary`, and the exact HTTP 409 mapping for
  `archive_run_recovery_required`; `archive_run_finalize_required` remains an
  internal CLI reason code because no public finalize endpoint is authorized; public
  create and decision request models and decision semantics remain unchanged
- `config/openapi_schema_snapshot.json` for the newly documented task component;
  the named property-level test, not the snapshot alone, proves selector exposure
- no changes to the owner-accepted design lock, EvidencePack contract, vision
  attestation, Gate-A runbook, workflow, test, manifest, any strict-JSON fixture
  below `tests/fixtures/review_reuse_er3/`, or contract verifier; all remain
  mode-`100644` and byte-identical by Git blob plus raw SHA-256, and the unchanged
  test reaches behavior supplied only by the authorized implementation files;
- `docs/development/L3_REVIEW_REUSE_ER3_GATE_B_IMPLEMENTATION_VERIFICATION_20260830.md`.

The branch must satisfy the complete ancestry chain:
`af72d0ec02b5d2dc1d92508539bc89ba857245a8 -> <ratified-design-lock-head> ->
<accepted-gate-a-head> -> <gate-b-implementation-head>`. Runtime implementation
may not start from a parallel branch or skip the accepted fail-first commit.
The accepted-Gate-A-to-implementation diff is closed as well: only the six
pre-existing runtime/config paths and four new files named above may change, with
status `M` limited to those existing paths with original mode/type preserved and
status `A` required exactly once as a mode-`100644` regular blob for all three new
modules plus the Gate-B verification document. Deletion, rename, copy, mode/type
change, symlink, submodule, Gate-A artifact drift, or any other path invalidates
Gate B.

The accepted, byte-identical Gate-A workflow has `name: ReviewReuse ER3 Contract`,
job id/name `er3-contract`, a
`pull_request` trigger with no `paths`/`paths-ignore` filter, `contents: read`,
`fetch-depth: 0`, and no `continue-on-error`, failure masking, optional step, or
shell suffix such as `|| true`. One gate step asserts the collected node-ID set is
exactly the 106 ratified names and that the run reports zero skipped, xfailed,
xpassed, deselected, or collection errors. Separate exact commands run the
canonical/digest verifier at
`tests/verification/review_reuse_er3_contract_verifier.py` and the frozen
ER1/ER2/v1 regression selection. Gate B receives the owner-named accepted head,
GitHub repository name/numeric ID, workflow blob, run/attempt/job/artifact IDs,
artifact archive digest,
strict report raw/full-canonical/self-excluding verification digests, frozen
candidate artifact-set digest, and protected accepted-evidence digest only through
an owner-protected CI environment consumed by the already-frozen workflow. Branch
files, PR fields, caller values, unrelated CI IDs, or the Markdown runbook cannot
substitute.

In Gate B, the accepted verifier runs from its exact Gate-A Git object outside the
implementation execution root, retrieves and byte-verifies that immutable evidence
artifact, and first uses authenticated GitHub API run, run-attempt jobs, and
artifact metadata to build the strict CI-lineage receipt. Repository name/ID,
accepted head, workflow path/blob, run/attempt, fixed `er3-contract` job, and
artifact must form one chain; the unexpired artifact must belong to that run and
match the downloaded archive digest. The frozen workflow has one evidence-producing
job and one digest-pinned upload step, closing the job-to-artifact join. Exact raw
API responses plus the normalized canonical lineage receipt are retained and
digest-bound. Every lineage value must also equal the report's protected producer
identity, closing cross-attempt artifact substitution. Syntactically valid
unrelated IDs fail. The verifier then proves the
full raw-object chain `runtime-base ->
design-lock -> Gate-A -> implementation`, validates the closed Gate-B path delta,
and compares exactly the eight singleton artifacts plus every additional strict-
JSON synthetic fixture
against the separate candidate artifact-set object frozen in the accepted Gate-A
report. The Gate-B comparison has its own digest and is not equated to the earlier
Gate-A artifact-set digest. It emits the strict
`review-reuse-er3-gate-b-verification-v1` report. Its exact-object-materialized,
closed-Python/import, fixture-denied, read-only implementation run must report
106/106 pass with zero
skip/xfail/xpass/deselect/collection/unclassified values. The
verifier hardcodes the owner-ratified canonical and raw-byte digests rather than
reading expected values only from each artifact's own self-digest field. Before
merge can ever be considered, repository branch protection must independently
show this exact check context as required; workflow YAML cannot prove that setting
and this document does not authorize merge.

The static substrate attestation and EvidencePack v2 contract are inputs to
implementation and remain byte-identical. Updating either, changing the fixture
manifest object from §4, or changing the pinned image requires a new design-lock
head and owner response.

The existing v1 files `dedup_live.py`, `dedup_adapter.py`, `evidence.py`, and
`canonical.py` remain byte-identical. If dispatch cannot be added without editing
one of them, implementation stops for a new owner review instead of widening the
write set.

The ER3 test file must scan the modified script and every new ER3 module, not only
`src/core` and API routes, for hosted-provider clients, ambient endpoint/store
selection, shell execution, retries/redirects, decision enablement, and unapproved
write paths. Existing broad static checks do not substitute for this named scope.

Any change outside this list, any auth/persistence/decision semantic change
beyond the explicit §5.4 version dispatch, or any attempt to combine ER4 requires
a new owner decision. No implementation PR may modify `eval_integrity_gate`,
training/feedback paths, assistant paths, cost-cap code, deployment, or PLM
write-back.

## 9. Acceptance and non-claims

ER3 is complete only when all of the following are true at one exact head:

- all 106 named fail-first/regression tests have a classified runtime-base baseline and are green at
  the implementation head;
- the owner-ratified static substrate attestation proves the exact image/source,
  mutable-path, configuration, and cardinality mechanisms, and the separately
  authorized runtime preflight and run receipts prove the observed zero-to-three
  transition;
- the runtime-preflight repository-state object proves a clean worktree at the
  exact owner-authorized implementation head and binds the reviewed command;
- the runtime manifest matches §4 digest
  `7fd1e774429fa5f75ab5728ed3e2a55b1972d18d8629da584bcf37dc1de91acf`;
- the v2 contract and golden digests remain
  `680c596d315061424a32d160d95131b7da9acc1a02a238397b2f83c40da8a372`
  and `422e23da3589b24a5539a3d6546cac98ba692046ea14930b179dd9f7fe1b9f7f`;
- existing ReviewReuse, identity, production-preflight, and core-fast suites are
  green without skips added for this tranche;
- an approved real private service run indexes the archive and searches the
  separate query without seeds, host publication, or an unknown candidate, and
  returns the exact receipt-digest-bound archive candidate;
- raw response, normalized receipt, and receipt-set digests match the v2 contract
  golden rules, and task context plus EvidencePack bind the exact set digest;
- score dimensions, missing evidence, confidence, and provenance satisfy §5;
- recorded export replay validates the summary and every artifact against the
  read-only persisted store without Docker/network access or any write;
- the development/verification document records exact SHA, commands, results,
  artifact digests, control-transport posture, and remaining limits;
- exact-head CI and independent review have no unresolved failing finding;
- the exact-head required check `ReviewReuse ER3 Contract / er3-contract` is green;
  unrelated synthetic/archive or host-published Docker checks do not substitute
  for this gate.

Even after those gates pass, the result proves only the controlled ER3 engine
path. It does not prove model quality, customer utility, production readiness,
decision authorization, deployment readiness, or pilot acceptance.

`evidence-pack-v2` remains an internal ER3-runner selection. This document does
not make it the default public create behavior and does not enable legacy v1 live
dedup. Any v2 default migration or v1 retirement is a separate owner decision.

## 10. Ratification texts

Suggested Gate-A fail-first-only owner response:

> I ratify `L3_REVIEW_REUSE_ER3_DESIGNLOCK_20260829.md` at exact head `<sha>` and
> bind runtime base `af72d0ec02b5d2dc1d92508539bc89ba857245a8`, manifest
> digest `7fd1e774429fa5f75ab5728ed3e2a55b1972d18d8629da584bcf37dc1de91acf`,
> image `ghcr.io/zensgit/dedupcad-vision@sha256:9f7f567e3b0c1c882f9a363f1b1cb095d30d9e9b184e582d6b19ec7446a86251`,
> static attestation
> `docs/development/L3_REVIEW_REUSE_ER3_VISION_SUBSTRATE_ATTESTATION_20260829.json`
> canonical/raw digests
> `4707f793f3597ebe12da4c6474bbb25208ccd9b1de48920a27c5ca083babe6cd` /
> `5d7e15aa82d714ad371e106deb3c6310cfc82c3062731392bfb3a6c2db175d94`,
> EvidencePack v2 contract
> `docs/development/L3_REVIEW_REUSE_ER3_EVIDENCE_PACK_V2_CONTRACT_20260829.json`
> canonical/raw digests
> `680c596d315061424a32d160d95131b7da9acc1a02a238397b2f83c40da8a372` /
> `9f30ee2c2f4e57180538ce493d28e21c7b8f8e75061cbafeabfad1a5153e1e96`,
> and golden EvidencePack digest
> `422e23da3589b24a5539a3d6546cac98ba692046ea14930b179dd9f7fe1b9f7f`.
> I authorize only §8.1 and its exact 106-node fail-first contract: the new test
> file, embedded-manifest copy, strict-JSON synthetic test fixtures, ER3 contract
> workflow, verifier, and Gate-A runbook. The branch must descend from the runtime
> base through this exact design-lock head.
> I do not authorize changes to `src/` or the existing runner, Docker/image access,
> fixture-byte reads, merge, decision enablement, deployment, pilot, or customer data.

This response authorizes producing a candidate Gate-A head and external evidence;
it does not accept either before the later Gate-B response names their exact
identities.

Suggested Gate-B mock-backed implementation owner response after reviewing the
exact Gate-A fail-first matrix:

> I accept Gate-A exact head `<gate-a-sha>` and authorize only §8.2 ER3 runtime
> implementation descending from `<design-lock-sha> -> <gate-a-sha>`. For repository
> `zensgit/cad-ml-platform` with numeric repository ID `<repository-id>`, I accept
> workflow
> `.github/workflows/review-reuse-er3-contract.yml` at blob `<workflow-blob>`, run
> `<run-id>` / attempt `<attempt>` / job `<job-id>` / artifact `<artifact-id>`, fixed
> artifact `review-reuse-er3-gate-a-evidence` with archive SHA-256
> `<artifact-sha256>`, and report `review-reuse-er3-gate-a-verification.json` with
> raw/full-canonical/self-excluding-verification SHA-256 `<report-raw>` /
> `<report-canonical>` / `<report-verification>`. I also accept candidate artifact-
> set SHA-256 `<gate-a-artifact-set>` and protected accepted-evidence SHA-256
> `<protected-input>`. Those exact values are loaded only through
> the owner-protected Gate-B CI environment. The implementation uses manifest
> digest `7fd1e774429fa5f75ab5728ed3e2a55b1972d18d8629da584bcf37dc1de91acf`, image
> `ghcr.io/zensgit/dedupcad-vision@sha256:9f7f567e3b0c1c882f9a363f1b1cb095d30d9e9b184e582d6b19ec7446a86251`, static attestation
> canonical/raw digests `<attestation-canonical>` / `<attestation-raw>`, EvidencePack
> v2 contract canonical/raw digests `<contract-canonical>` / `<contract-raw>`, and
> golden EvidencePack digest `<golden-pack>`. Verification remains mock-backed. I do
> not authorize a Docker connection, image pull/start, fixture-byte processing,
> merge, decision enablement, deployment, pilot, or customer data.

Suggested Gate-C repository-fixture-run owner response after exact-head Gate-B
implementation review. This response is an out-of-band human governance artifact;
`--authorized-head` and the other CLI arguments enforce integrity only and cannot
authenticate that the owner issued this response:

> I authorize one ER3 repository-fixture run at implementation exact head `<sha>`
> using only the ratified manifest/image/contracts; absolute Python interpreter,
> dependency root, and `python_control_sha256=<python-control>` under `-I -S`;
> Docker source binary `<canonical-path>` with parent-chain digest
> `<binary-parent-sha256>` and SHA-256 `<binary-sha256>` copied to the sealed
> private executable; endpoint `<absolute-unix-uri>` with approved socket parent-
> chain and `socket_identity_sha256=<socket-sha256>`; run ID
> `rr-er3-<32-hex>` and resource-owner ID `<64-hex>`; Docker Engine API `1.43`,
> platform `<linux/amd64|linux/arm64>`, and exact reviewed-command digest
> `<command-digest>`.
> The runner must stop before opening fixture bytes unless §5.2 preflight matches
> every bound value and must stop on any socket/parent-chain drift. I do not
> authorize merge, decision enablement, deployment, pilot, customer data, or any
> other image/service.

Until Gate A names the exact document head and base, no test/workflow tranche may
start. Until Gate B names the accepted candidate-Gate-A exact head, immutable CI
artifact identity, report raw/full-canonical/self-excluding verification digests,
frozen candidate artifact-set digest, protected-input digest, and every design/
runtime artifact digest, no runtime implementation may start. Until
Gate C names the implementation exact
head, Python/import control, source/private binary, endpoint/socket identities,
run/resource-owner IDs, API version, platform, and command digest,
the image must not be pulled or started and fixture bytes must not be processed.
