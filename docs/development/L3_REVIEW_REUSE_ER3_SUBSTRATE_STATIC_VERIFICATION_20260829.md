# ReviewReuse ER3 Substrate Static Verification

**Status:** STATIC VERIFIED / RUNTIME UNVERIFIED / DOCS ONLY

**Date:** 2026-08-30

**CAD ML base:** open PR #584 exact head
`af72d0ec02b5d2dc1d92508539bc89ba857245a8`

**Vision source:** `https://github.com/zensgit/dedupcad-vision.git` exact revision
`2fc35d60ff034c9f790868c02381a9716becc942`

**Runtime authority:** NONE

This record verifies static OCI metadata, exact source behavior, repository
fixtures, and canonical digests used by the ER3 design-lock proposal. It does not
authorize or claim image pull, container start, fixture processing, ER3 runtime
implementation, merge, decision enablement, deployment, pilot, or customer data.

## 1. Artifacts verified

| Artifact | Value |
|---|---|
| Design lock | `L3_REVIEW_REUSE_ER3_DESIGNLOCK_20260829.md` |
| Static attestation | `L3_REVIEW_REUSE_ER3_VISION_SUBSTRATE_ATTESTATION_20260829.json` |
| EvidencePack v2 contract | `L3_REVIEW_REUSE_ER3_EVIDENCE_PACK_V2_CONTRACT_20260829.json` |
| Embedded manifest proposal canonical digest; no manifest file exists at this docs-only head | `7fd1e774429fa5f75ab5728ed3e2a55b1972d18d8629da584bcf37dc1de91acf` |
| Static attestation digest | `2d32ff91eb0f42079541f85dbc926f144c13bf1cbbe8ba2b720164fe4329f28e` |
| Static attestation raw file SHA-256 | `23680e3625c6d87d083998e8cdcfdb9004dd19350db81755c133f4eeb88ceb68` |
| EvidencePack v2 contract digest | `59f51f4a4b442c81a0431135291a44f41f178335b0de3e615245e0e01e8224e9` |
| EvidencePack v2 contract raw file SHA-256 | `87ecaf430438e52fda7c8e8f3c4abc0f9eac07388306f6fcdcb7fa48f197d158` |
| Embedded EvidencePack golden digest | `b012ad952a83b4ed29b402818440b364985613a20199c54389e3bc5e5c91c104` |
| Verified fixture-set golden digest | `536113fa48ac2f692635959bd5f1d0f8ac92faff1fc8ab9f27679a5f474e51b1` |
| Vision OCI index | `sha256:9f7f567e3b0c1c882f9a363f1b1cb095d30d9e9b184e582d6b19ec7446a86251` |

The manifest, attestation, contract, and embedded golden digests were recomputed
with `src.core.review_reuse.canonical.canonical_sha256()` after removing only
their respective self-digest field. Strict I-JSON parsing passed for both JSON
artifacts. Raw file SHA-256 values separately pin the reviewed bytes so a test
cannot trust only a self-reported canonical digest. The golden vectors are serialization and digest vectors, not model-
quality or runtime evidence. The critical provenance vectors are:

| Canonical preimage | Golden SHA-256 |
|---|---|
| Post-index search response receipt | `b4ff738ada27842e09ff22791b52cc504860f589d23e6cbcc18bc9e542975602` |
| Service identity | `b9defaaa4689ea63652663cc5431c8c431a23a89d31f4f0227908ee49278660e` |
| Score-mapping ruleset | `ccee15b504054a1cd3def3f6531babbc41a72742d3d2dbb8e35efd57337afd11` |
| Python/import control | `586aedad7a722501f751b48926fb757464058bb3213d9b9994deae967c238915` |
| Reviewed command | `2f0394cef55cec77dfaca3aa20c5f6c8501d0827080b8fc19a3238d76702bf91` |
| Docker socket identity | `494b331e50b32565a2d20eda74afc69a5cb698e53be47363701046b9ab80d850` |
| Raw HostConfig v1.43 | `9ab001ac45b9fdbedb61dc2180864659298ed5acda1f28333a8196824cea3bae` |
| Raw Config.Env | `cad77eb9bd322cb3082639d8e287f488afc85d2e03d2cb56922698cf63018a19` |
| Resource ownership | `103acb5087f3624731243d89dc1cb210dfd2c039d0ac24894def8acd28dbcd9d` |
| Docker control plane | `ef6802d6138b19db4f580a75ac83fdf99b6d04e2bd3c56199d4e0484ca6c5ea7` |
| Exact Docker child-environment artifact | `571117807ac16fa31fd8cb124a55a2263f050bb04db25fb94b08774f6b53d51d` |
| Final run-journal artifact | `615ec0ac25a4ca8edd83bacc589d4f411390ab544e3a9c71e2e625a85f547031` |
| Strict runtime inspect projection | `c907c2aa218065321eba9ac471a4436286f363632809fcbd8c15a8c6838cf927` |
| Runtime preflight | `c9290dde847d946d96e1212740bac982bb764279b26fe5712cf35494b9c48e21` |
| Continuous runtime seal | `939bc5e9334738aa249af8c9825d7c5c9b94b7f8a0852cde4ec379a23e4da500` |
| EvidencePack v2 Markdown bytes | `c91711621d0853f113c701f313c47e8cf3f229900b498f04a9bb941db36d760e` |
| Success-intent golden vector | `1542e9a7ecaca4edeb7bc06b210c4ab0d91923a5f547c2be1bc8aaf97e68dfc4` |
| Run-summary golden vector | `2c30944a625ec5002114ea716239598f5a7db5f67606301de2797f0e4512d5fc` |

## 2. OCI and source binding

Authenticated read-only GHCR registry inspection resolved the pinned OCI index
to these platform objects:

| Platform | Manifest | Config / expected Docker image ID | SLSA statement layer |
|---|---|---|---|
| `linux/amd64` | `sha256:036db4725b5578689d76c1ced933aa9438bd628e257387515e214de5a838ce20` | `sha256:9d056adc789ee0980de046205f0bb009c57a36a5a94ea329c2971e5cecf30bc0` | `sha256:1fd52fd641704c20d678c4133741a3800e6cf0f1b84b3e98c150cb4dc82517a0` |
| `linux/arm64` | `sha256:c81c77d8672aa36854e9cf0d650f689958266d8b86297abf1c39358038e84689` | `sha256:6895fa7587fa7e1e541e3eb0f440bef94715e631d8a116a3ab2e7b3dcd76a838` | `sha256:17910b237e2d192e8ef8423c26cf129340a8eb8019ae128801e8df60da8cb59b` |

Both SLSA v0.2 statements bind the image to the source repository and revision
named above. They report subject version `1.1.1`, `reproducible=false`, and
incomplete materials. Exact source reports `1.1.0` in `pyproject.toml`, while
health/OpenAPI report `0.2.0`. The design therefore treats only the image digest
and source revision as identity authority.

Exact source evidence:

- `Dockerfile:48-68` copies only source/package metadata, runs as `appuser`, and
  declares `/app/data`, `/app/indexes`, and `/app/logs` volumes.
- `Dockerfile:37-42,70-72` installs curl in the runtime stage and uses it for the
  image health check. Runtime preflight must still prove `curl --version` succeeds
  before accepting the bounded no-egress probe result.
- `Dockerfile:77-78` starts Uvicorn on container port 8000.
- `pyproject.toml:5-49` identifies package version and image-processing
  dependencies.

## 3. Input media correction

The original draft selected DXF files. Exact source disproves that contract:

- `api/routes_index.py:237-358` sends index uploads directly to
  `PerceptualHash.compute_hash()` and `FeatureExtractor.extract()`. Its signature
  also requires `user_name: str = Query(...)`; omitting `user_name` produces 422
  before any index receipt.
- `api/routes_progressive.py:111-225` sends search uploads directly to the
  progressive image engine.
- `core/phash.py:60-127` and `core/features.py:179-276` load through pyvips,
  Pillow, or OpenCV; neither path invokes ezdxf.

DXF-specific endpoints exist elsewhere, but they are not the index/search path.
The proposal was corrected to use deterministic tracked PNGs:

| Fixture | Bytes | SHA-256 |
|---|---:|---|
| `tests/vision/fixtures/cad_features/cad_line.png` | 235 | `e32de24b2a39e9ee56932c80fc5f28497c01ff36348ee2cb0e363d750ae2334c` |
| `tests/vision/fixtures/cad_features/cad_circle.png` | 368 | `c32bd82c54124d87fe0731f7cc1a05e4d0147d755d9cd98dc41a1aaa047495e6` |
| `tests/vision/fixtures/cad_features/cad_arc.png` | 493 | `9a9ca171ef968202eb69523eb0db9e8f88009cb0048a0d301dfa87076bb48d7e` |

The query is a separately named logical request using the tracked line bytes.
Only the archive role may call index-add.

The corrected exact index request is
`POST /api/index/add?user_name=review-reuse-er3-fixture&upload_to_s3=false` with
multipart field `file`. The attestation now freezes that URL and every health,
stats, rebuild, pre-index search, and product-search curl argv suffix. Redirects,
retries, Unix sockets, proxies, and user-supplied HTTP fragments are forbidden.

## 4. State, cardinality, and rebuild

Exact source inspection established:

- `storage/factory.py:36-49` selects filesystem mode and `VISION_CACHE_DIR`.
- `storage/filesystem_backend.py:25-42,101-151,250-276` stores drawing JSON/NumPy
  records and exposes `total_drawings` as the in-memory drawing-map length.
- `api/routes_index.py:392-406` exposes that count at `GET /api/stats`.
- `search/phash_index.py:60-73,240-278` uses `./data/phash_index.pkl`.
- `search/faiss_index.py:37-59,234-309` uses `./data/faiss_index/` with
  `index.faiss` and `id_mapping.pkl`.
- `api/routes_license.py:43-63` and `api/routes_version.py:118-147` create SQLite
  files under `./data` during module import.
- `storage/s3_backend.py:483-494,666-721` initializes local fallback storage at
  `/tmp/dedupcad-storage`, even when index-add later receives
  `upload_to_s3=false`.

Consequently `/app/data`, `/app/indexes`, `/app/logs`, and `/tmp` must all be
explicit run-scoped tmpfs mounts over a read-only root. `/dev/shm` must remain a
private non-host-bound tmpfs with private IPC. Explicit tmpfs overrides also
prevent image-declared anonymous volumes.

The revised control-plane contract rejects every ambient Docker
endpoint/context/TLS/config/API/platform override and never reads an active/default
context. Gate C must instead bind an exact absolute `unix://` endpoint, socket and
parent-chain identity, API version, platform, binary, and reviewed command. Its
API-v1.43 raw HostConfig key contract plus normalized projection closes privilege,
resources, host namespaces, devices, binds, Docker/socket publication, ports,
extra hosts, and restart policy. Because the daemon is shared administrative
state, every copy/exec is bracketed by before/after inspect projections and raw
stream artifacts bound into a continuous runtime-seal receipt.

With `EVENT_BUS_ENABLED=false`, `events/event_bus.py:700-729` selects an in-memory
LocalEventBus and does not create Redis. Index-add still publishes to that local
bus, but startup did not register index handlers. Therefore
`POST /api/v2/index/rebuild` is mandatory.
`orchestrator/progressive_engine.py:593-612` rebuilds L1/L2 from filesystem
storage. The required observed sequence is:

1. storage count 0 and L1/L2 sizes 0;
2. three successful unique index receipts;
3. rebuild success;
4. storage count 3 and L1/L2 sizes 3/3;
5. only then, the product query.

The service has no drawing-list endpoint at this revision. Archive identity is
therefore proven jointly by the zero-count precondition, exact three-receipt hash
set, distinct drawing IDs, and post-count three. L1/L2 sizes are required
readiness evidence, not the identity authority.

## 5. Search and EvidencePack constraints

`api/routes_progressive.py:34-37,156-173,221-223` caches search responses for five
minutes by file MD5 and search options. Reusing identical options for the empty
probe and post-index query would replay a stale zero result. The design now pins
different option tuples and requires a discriminator test.

`api/routes_progressive.py:605-634` exposes provider aggregate fields plus explicit
L1-L4 fields. `orchestrator/models.py:71-165` shows that top-level similarity and
confidence are provider aggregates. The v2 mapping therefore uses only
`levels.l2.feature_similarity` as normalized visual evidence, keeps L1 as method
evidence, emits null semantic/geometric values for disabled L3/L4, and does not
relabel provider aggregate similarity/confidence as calibrated product values.

The first design draft still left four implementation choices underspecified:

- the exact v2 object keys, types, nullability, ordering, digest exclusions, and
  legacy v1 boundary;
- how a manifest-bound PNG enters the internal ER3 flow while the public
  `ReviewReuseService.create_task()` and HTTP create route correctly remain
  DXF-only;
- which persisted record is the replay fact source and how every exported
  artifact is verified without a second network call or store write;
- how task responses and EvidencePack responses prove that the immutable schema
  selector matches the stored pack.

The added v2 contract closes those choices with recursively forbidden unknown
fields, a persisted context schema, selector-conditioned calibration, a golden
digest vector, and exact task/pack/provenance invariants. The design keeps public
create on v1 and proposes one internal manifest-bound PNG seam only. Replay starts
from the read-only store after verifying the externally bound summary and every
artifact digest. Dedicated API response models, a discriminated v1/v2 pack union,
and property-level OpenAPI tests are mandatory; a generic dictionary response or
snapshot-only check is insufficient.

Source review also confirmed that the fixed vision service has no authentication
dependency on index-add or rebuild. A host-loopback listener would therefore
leave a mutation surface available to other host processes. The corrected
contract uses Docker network mode `none`, publishes no TCP or Unix socket, binds
Uvicorn to in-container loopback, and performs only fixed-argv `docker exec` and
`docker cp` through the local Docker daemon. It additionally requires the post-
index result to include the exact `(drawing_id, file_hash,
index_receipt_sha256)` triple bound by the `archive-exact-001` canonical receipt;
a zero-hit, unknown-candidate, or receipt-digest mismatch is failure even when the
provider reports `success=true`.

An exact-head GitHub Codex review of `0b7ca4975d2ccc57890e6e408bccff873ca99712`
then found two remaining contract defects:

- candidate provenance matched only receipt identity, not the digest of that
  exact receipt, while the receipt-set canonical preimage and ordering were not
  frozen;
- accepting curl timeout exit 28 could pass when outbound networking existed but
  the fixed target silently dropped packets.

The subsequent docs-only revision closes both. The v2 contract now freezes the
strict six-field supplier response, normalized receipt, manifest-ordered receipt
set, their exclusions and golden vectors, plus candidate/context/pack equality.
The no-egress probe accepts only curl exit 7 with HTTP code `000`; exit 28 fails,
and Docker `NetworkMode=none` with no attachments remains authoritative. These are
design corrections, not runtime observations.

A later read-only multi-model review of PR #585 head `18662c895...`, followed by
direct source verification, found another pre-ratification tranche:

- the pinned index-add route requires `user_name`, so the documented request
  would have failed with 422;
- search, service-identity, mapping-ruleset, and runtime-preflight SHA fields had
  no exact canonical preimages, while the golden pack used repeated-character and
  all-zero placeholders;
- the local Docker TCB did not reject remote contexts, close security-critical
  HostConfig fields, classify `/dev/shm`, or detect post-preflight posture drift;
- ambient host store/provider/proxy/output inputs were not quarantined;
- query bytes were described as one-shot while the protocol searches twice;
- mapped candidates existed before `recall_started`, making the proposed event
  chronology false, and replay had no unambiguous location in the write set;
- `allowed_actions` confused a decision vocabulary with enablement, and the fixed
  visual-only mapping was not canonicalized separately from supplier verdicts;
- no dedicated exact ER3 CI gate was locked.

This revision closes those design defects with fixed query/curl argv, local-unix
Docker resolution, a closed HostConfig projection, operation-by-operation seal,
host-environment quarantine, two explicit query transfer pairs, a service-
bracketed archive controller lifecycle, explicit script replay subcommand,
closed vocabularies, canonical provenance receipt/ruleset preimages, and a
named ER3 CI gate. They remain proposed contracts, not runtime observations.

A focused local read-only review of that expanded draft then confirmed five more
contract defects before ratification:

- fixture hash validation and preflight both claimed to occur before the first
  fixture-byte read, an impossible ordering;
- the continuous inspect preimage omitted the image, command, process user,
  lifecycle, network attachment, and published-port fields claimed by prose;
- runtime evidence accepted any syntactically valid repository SHA and did not
  prove a clean owner-authorized head or reviewed command;
- query bytes could change between one host hash and either later transfer;
- requiring an explicit legacy/demo subcommand would break the existing bare
  Makefile/test invocations outside the locked write set.

This revision closes those design defects by checking manifest structure before
Docker, opening and hashing fixture bytes only after preflight, transferring only
immutable verified buffers, binding each copy digest, expanding the strict inspect
projection, binding clean exact-head/command evidence into preflight, and preserving
the bare synthetic invocation's argument/control flow plus frozen v1 EvidencePack
while making both ER3 modes explicit and non-fallback. It also aligns cleanup failure semantics and the owner
ratification paths. These remain design contracts, not runtime observations.

The first clean exact-head follow-up at `f6f6db05a381e8e6135ebaa469bad223a09e34dd`
then found three remaining P2 contract gaps: the inspect projection omitted `/app`
and eleven attested environment values/absence markers; §1 counted six gate items
while §10 required seven; and whole-output byte compatibility was impossible once
legacy tasks gained an explicit v1 selector. A parallel Sonnet review also found
that the fail-first gate did not explicitly authorize pure Docker subprocess mocks.
This revision closes all four without authorizing a daemon connection: the strict
projection now binds the complete attested environment and working directory, the
gate count is seven, legacy compatibility is narrowly stated, and §7 permits only
mocked Docker CLI/inspect responses before the then-proposed runtime gate.

A later exact-source review after #584 advanced to `af72d0ec02b5...` found a
second set of pre-ratification gaps: `input_validated` could not be truthful while
fixture reads remained post-preflight; `recall_started` was not durably persisted
before side effects; interrupted runs had no non-reexecuting recovery protocol;
the Docker seal bound repeated projections rather than full command and fresh
inspection receipts; Docker client configuration and binary identity were not
quarantined; security-relevant HostConfig values were omitted; and run summary,
writer lease, artifact inventory, and v2 Markdown bytes were not closed schemas.

The current docs-only revision addresses those findings with a v2-only four-
revision protocol, explicit failure-only `er3-recover`, secure one-descriptor
fixture reads, complete Docker command/inspection receipt chains, expanded closed
HostConfig, exact run-summary inventory, and deterministic Markdown. It also binds
the required ancestry as #584 runtime base -> ratified design-lock head ->
implementation head. These are proposed contracts only. No ER3 test, runtime
module, Docker operation, runtime fixture transfer/decoding/index/search, or owner
action was performed. The static verification read the three tracked fixture files
only to compute the sizes and SHA-256 values recorded in §3 and to reconstruct the
deterministic in-memory ustar serialization vectors. It did not decode an image,
load a model, build an index, call search, or expose fixture bytes to a runtime.

An exact-head Sol/Terra/Luna review of `657b27d1e294c8ca85d4cd4fb000e959c84fcd05`
then rejected ratification because running tasks had no revision-1 run binding,
active cancel could escape recovery, fail-first and runtime authority were combined,
local-unix context resolution did not bind a trusted socket, raw HostConfig mapping
and receipt artifacts were incomplete, ancestor-directory TOCTOU remained, revision
3 bound only one observed hash, probe argv conflicted, and strict-verification/
manifest wording was ambiguous. The current revision responds by splitting Gates
A/B/C; adding immutable `ER3RunBinding`, cancel/recovery constraints, explicit
endpoint/socket/API binding, API-v1.43 raw mapping, directory-FD opens, complete
verified-fixture-set and replay artifacts, and corrected documentation. These are
still proposed docs-only contracts and require another exact-head review.

The next exact-head read-only review of `aa11654dd07ec4f53a058ed76a93f230763a6e11`
also returned `REQUEST_CHANGES`. Direct source and contract checks confirmed that
the existing-root classifier was not mutually exclusive, recovery had no durable
command-prefix journal, the run lease could be unlinked and recreated, the closed
store inventory and terminal canceled/failed states were incomplete, the Docker
child-environment preimage and per-command binary/socket identities were not
artifact-bound, setup/operation vectors still admitted synthetic argv, and the
106 Gate-A nodes lacked a sole machine classifier with an exact baseline map.
This working revision closes those design defects, including permanent run/store
lease inode rules and out-of-band Gate-C authority. It still requires a fresh
review of the new exact head; this paragraph is not ratification.

## 6. Commands and results

The static verification used read-only commands equivalent to:

```bash
git show 2fc35d60ff034c9f790868c02381a9716becc942:<source-path>
git grep -n <pattern> 2fc35d60ff034c9f790868c02381a9716becc942 -- src Dockerfile pyproject.toml
sha256sum tests/vision/fixtures/cad_features/cad_line.png \
  tests/vision/fixtures/cad_features/cad_circle.png \
  tests/vision/fixtures/cad_features/cad_arc.png
/opt/homebrew/bin/python3.11 - <<'PY'
from pathlib import Path
from src.core.review_reuse.canonical import strict_json_loads
for path in (
    Path("docs/development/L3_REVIEW_REUSE_ER3_VISION_SUBSTRATE_ATTESTATION_20260829.json"),
    Path("docs/development/L3_REVIEW_REUSE_ER3_EVIDENCE_PACK_V2_CONTRACT_20260829.json"),
):
    strict_json_loads(path.read_bytes())
PY
/opt/homebrew/bin/python3.11 -m pytest -q \
  tests/unit/test_review_reuse_canonical.py \
  tests/unit/test_review_reuse_evidence_goldens.py \
  tests/unit/test_review_reuse_er1_store_integrity.py
/opt/homebrew/bin/python3.11 -m pytest -q tests/unit/test_review_reuse_*.py
git diff --check
```

Results:

- fixture byte sizes and SHA-256 values matched the embedded manifest proposal;
- embedded manifest proposal canonical digest matched; no physical manifest file
  or raw manifest-file SHA is claimed at this docs-only head;
- static attestation strict parse and canonical digest matched;
- v2 contract strict parse, canonical digest, and embedded golden digest matched;
- all three raw-response, normalized-receipt, and receipt-set golden vectors
  recomputed to their stored digests;
- search-response, service-identity, score-ruleset, Docker-control, strict
  inspect, repository-bound runtime-preflight, 22 distinct operation-command
  receipts, 44 distinct inspection receipts, cleanup-command receipts,
  continuous seal, EvidencePack, deterministic Markdown, run summary, and
  whole-contract golden preimages recomputed to their stored digests;
- a read-only cross-artifact verifier passed `30/30` closure categories, including
  contract/attestation self digests, exact closed store/control/environment/journal
  inventory, persistent lease semantics, mutually exclusive root classification,
  106-entry Gate-A matrix with the exact 104-red/2-existing-pass split, and every
  downstream EvidencePack/intent/summary digest;
- a separate machine-readable receipt derivation passed `79/79`: all nine setup,
  22 operation, 44 fresh-inspection, and four cleanup receipts recomputed from
  their complete ten-field preimages. Setup and operation receipts now use exact
  real Docker argv matrices and each receipt binds fresh private-binary/socket
  identities; synthetic stream bytes remain serialization vectors only;
- source paths and endpoint semantics above matched exact revision `2fc35d60...`;
- the design names 106 unique fail-first/regression tests, fixes tests 28 and 34 as
  the only existing-pass baseline nodes, maps every other node to a named missing-
  behavior boundary, and preserves public DXF/v1 behavior;
- the recovery classifier now has mutually exclusive initialization, terminal,
  resumable, recovery, finalize, complete, and mismatch branches; only a durable
  run-journal plus exact command-artifact prefix can authorize cleanup;
- the filesystem store is a closed one-tenant/one-task/optional-index inventory;
  its permanent writer-lease path and locked FD must retain the same validated
  inode while the export-freeze remains held through summary parent fsync;
- the existing canonical, EvidencePack golden, and ER1 store-integrity regression
  selection passed `121/121` under Python 3.11;
- the complete existing ReviewReuse unit selection passed `222/222` under Python
  3.11; only seven pre-existing ezdxf/pyparsing deprecation warnings were emitted;
- the current docs-only PR checks are not the future Gate-A fail-first workflow and
  provide no evidence that the proposed 106-node matrix has been implemented;
- no runtime code was modified by this static tranche.

An initial invocation through `/usr/bin/python3` used Python 3.9.6 and failed in
test setup while FastAPI evaluated a repository `str | None` annotation. It did
not execute the selected tests. The repository CI uses Python 3.10/3.11; the
recorded result above is the rerun through the installed Python 3.11 interpreter.

## 7. Explicitly unverified

`docker info` failed with:

```text
Cannot connect to the Docker daemon at unix:///Users/chouhua/.docker/run/docker.sock.
Is the docker daemon running?
```

Therefore none of the following is claimed:

- selected runtime platform or actual Docker image ID;
- Gate-C Docker endpoint/socket locality, API version, or server identity;
- read-only root, four explicit tmpfs mounts, private `/dev/shm`, no anonymous/
  host volume, closed HostConfig, or process-user/command/workdir/environment posture;
- Docker network mode `none`, absence of additional attachments or host-published
  TCP/Unix sockets, or the in-container loopback bind;
- fixed-argv Docker control execution, freshly observed per-command identities,
  operation-by-operation inspect seal, curl availability
  plus the exact numeric bounded no-egress probe exit/status contract, or the
  local-unix Docker-daemon administrative boundary;
- observed zero/three counts or L1/L2 sizes;
- image startup, PNG decoding, real index receipts or receipt-set digests, rebuild,
  search, cleanup, or replay.

Across earlier snapshots, Sol, Terra, Luna, and the Claude CLI models supplied
advisory reviews. On `aa11654d...`, Sol/Terra/Luna and Sonnet returned results;
Opus and Fable did not run because the configured CLI budget was insufficient.
Kimi K3 returned a weekly-quota 403 and supplied no result. Grok 4.6 reached
source inspection but ended in provider network failure and supplied no final
finding. Failed or unavailable invocations are not counted as reviews.
Direct source checks, not model agreement alone, drive corrections. Both
`657b27d1...` and `aa11654d...` were `REQUEST_CHANGES`, not approvals; a separate
read-only review of the next exact head is required before ratification. None of
these model reviews is runtime verification.

## 8. Static conclusion

The fixed image is a plausible ER3 visual substrate only for manifest-bound PNG
input and only under the corrected isolation, rebuild, cache, canonical receipt,
exact-candidate, score-mapping, versioned EvidencePack, and separate
implementation/runtime owner gates. The static attestation and v2 contract are
ready for review but do not satisfy runtime preflight and grant neither ER3
implementation nor run authority.
