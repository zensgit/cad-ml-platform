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
| Archive manifest digest | `7fd1e774429fa5f75ab5728ed3e2a55b1972d18d8629da584bcf37dc1de91acf` |
| Static attestation digest | `988138a75031d78916ad388e1580012a50f8ac9bf5b97a9461a67d9ed2231228` |
| Static attestation raw file SHA-256 | `76ab3d92eafdb386f6f3588308da4b1e45d4830fb902fb1b6059a43df86a865c` |
| EvidencePack v2 contract digest | `2d5039d261f63d3e5db2d8da0579184540886871899641d1e4f816418c22accd` |
| EvidencePack v2 contract raw file SHA-256 | `29f14b41971b728bf0c8481e66641792749c2d72b333bfc2e34fe364c687fd22` |
| Embedded EvidencePack golden digest | `96891fe5d32cd3ed65be22bb741e2d935240a127823e6a74215b6c7d2a04329d` |
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
| Docker control plane | `5e89a5509a6c04f857be74421e835495331537596dfdfa26547b4f9b6cdc70f4` |
| Strict runtime inspect projection | `e005d813db825019486269b41452d48de8ebbc7872f41f9336f256ba77f31a20` |
| Runtime preflight | `84bd2e9461baf20ed04102533a04464625f1a524b14477a65f3d43688ceef541` |
| Continuous runtime seal | `d55c2d781d03f8b267b65fefd1cc2a37fe296d411567b255bc3c5c3d0a0a1230` |
| EvidencePack v2 Markdown bytes | `1410bdc7cf1953c569ee16459724d83e866d04f5bfbd64b62c0477186c7a2460` |
| Run-summary golden vector | `9f18176f0052ebf5f52310adc253f152c9b9c52150861bf25a15132e33a645dd` |

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

The revised control-plane contract rejects every Docker endpoint/context/TLS
environment override and accepts only an active context resolved to a local
absolute `unix://` URI. Its normalized HostConfig projection closes privilege,
capabilities, host namespaces, devices, binds, Docker/socket publication, ports,
extra hosts, and restart policy. Because the daemon is shared administrative
state, every copy/exec is bracketed by before/after inspect projections and bound
into a continuous runtime-seal receipt; a one-time preflight is insufficient.

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
closed vocabularies, five canonical provenance receipt/ruleset preimages, and a
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
mocked Docker CLI/inspect responses before the second owner gate.

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
module, Docker operation, fixture read, or owner action was performed.

## 6. Commands and results

The static verification used read-only commands equivalent to:

```bash
git show 2fc35d60ff034c9f790868c02381a9716becc942:<source-path>
git grep -n <pattern> 2fc35d60ff034c9f790868c02381a9716becc942 -- src Dockerfile pyproject.toml
sha256sum tests/vision/fixtures/cad_features/cad_line.png \
  tests/vision/fixtures/cad_features/cad_circle.png \
  tests/vision/fixtures/cad_features/cad_arc.png
python3 -m json.tool \
  docs/development/L3_REVIEW_REUSE_ER3_VISION_SUBSTRATE_ATTESTATION_20260829.json
python3 -m json.tool \
  docs/development/L3_REVIEW_REUSE_ER3_EVIDENCE_PACK_V2_CONTRACT_20260829.json
/opt/homebrew/bin/python3.11 -m pytest -q \
  tests/unit/test_review_reuse_canonical.py \
  tests/unit/test_review_reuse_evidence_goldens.py \
  tests/unit/test_review_reuse_er1_store_integrity.py
/opt/homebrew/bin/python3.11 -m pytest -q tests/unit/test_review_reuse_*.py
git diff --check
```

Results:

- fixture byte sizes and SHA-256 values matched the proposed manifest;
- manifest canonical digest matched;
- static attestation strict parse and canonical digest matched;
- v2 contract strict parse, canonical digest, and embedded golden digest matched;
- all three raw-response, normalized-receipt, and receipt-set golden vectors
  recomputed to their stored digests;
- search-response, service-identity, score-ruleset, Docker-control, strict
  inspect, repository-bound runtime-preflight, 22 distinct operation-command
  receipts, 44 distinct inspection receipts, cleanup-command receipts,
  continuous seal, EvidencePack, deterministic Markdown, run summary, and
  whole-contract golden preimages recomputed to their stored digests;
- source paths and endpoint semantics above matched exact revision `2fc35d60...`;
- the design names 89 unique fail-first/regression tests and preserves public DXF/v1 behavior;
- the existing canonical, EvidencePack golden, and ER1 store-integrity regression
  selection passed `121/121` under Python 3.11;
- the complete existing ReviewReuse unit selection passed `222/222` under Python
  3.11; only seven pre-existing ezdxf/pyparsing deprecation warnings were emitted;
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
- active Docker context/endpoint locality or server identity;
- read-only root, four explicit tmpfs mounts, private `/dev/shm`, no anonymous/
  host volume, closed HostConfig, or process-user/command/workdir/environment posture;
- Docker network mode `none`, absence of additional attachments or host-published
  TCP/Unix sockets, or the in-container loopback bind;
- fixed-argv Docker control, operation-by-operation inspect seal, curl availability
  plus the exact numeric bounded no-egress probe exit/status contract, or the
  local-unix Docker-daemon administrative boundary;
- observed zero/three counts or L1/L2 sizes;
- image startup, PNG decoding, real index receipts or receipt-set digests, rebuild,
  search, cleanup, or replay.

The focused local round completed through Sol, Terra, Luna, and the Claude CLI
canonical models `claude-opus-5`, `claude-sonnet-5`, and `claude-fable-5`. Kimi
K3 returned a weekly-quota 403 and supplied no result. Grok 4.6 reached source
inspection but remained in provider retry and supplied no final finding, so it
is not counted as a completed review.
Direct source checks, not model agreement alone, confirmed every correction
applied in this revision. A separate read-only review of the new exact head is
still required before ratification. These reviews are design evidence only, not
runtime verification.

## 8. Static conclusion

The fixed image is a plausible ER3 visual substrate only for manifest-bound PNG
input and only under the corrected isolation, rebuild, cache, canonical receipt,
exact-candidate, score-mapping, versioned EvidencePack, and separate
implementation/runtime owner gates. The static attestation and v2 contract are
ready for review but do not satisfy runtime preflight and grant neither ER3
implementation nor run authority.
