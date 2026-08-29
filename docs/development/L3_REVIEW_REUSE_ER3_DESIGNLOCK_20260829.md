# ReviewReuse ER3 Real Isolated Archive Design Lock

**Status:** FOR REVIEW / NOT RATIFIED / DOCS ONLY

**Date:** 2026-08-29

**Authority:** `docs/PRODUCT_STRATEGY.md` §3.3 and the owner-ratified #583 exact head
`9150e06c75721bf086572ed271b68548104e8300`

**Draft base:** open PR #584 exact head
`13055d4966fb3b543d747759ebd237d24a55e452`

**Runtime authority:** NONE

This document proposes the ER3 contract. It does not authorize implementation,
merge, decision enablement, deployment, pilot activity, or customer drawing use.
The current owner authorization covers ER1+ER2 only.

## 1. Decision requested

ER3 implementation ratification must name all six of the following:

1. the exact implementation base SHA;
2. the exact repository-fixture manifest from §4;
3. the digest-pinned private vision image from §5.2;
4. the static substrate attestation exact path and digest from §5.2;
5. the exact EvidencePack v2 contract path and digest from §5.4;
6. the fail-first contract in §7.

Without all six, ER3 implementation remains blocked. Ratification of this
document authorizes only fail-first implementation and mock-backed verification.
It does not authorize pulling or starting the vision image, sending fixture
bytes, merging any PR, setting `REVIEW_REUSE_DECISIONS_ENABLED`, deployment,
pilot use, or processing customer data. A second owner response must authorize
the repository-fixture run at an exact implementation head after the runner and
its command are reviewable.

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

These observations were reproduced against draft base `13055d4966...`:

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
field itself. Entry IDs and run-scoped names must be unique. Duplicate hashes
inside the archive role fail. A source path and hash may occur once in an archive
entry and once in the query entry only when the query declares that archive as a
byte-identical fixture relationship. The query logical role and run-scoped name
are never sent to index-add. Path escape, symlink escape, missing files,
size/hash drift, unknown fields, or multiple query entries fail before any
network call.

## 5. Runtime contract

### 5.1 Entrypoint and isolation

- The documented direct command works from a clean checkout with `PYTHONPATH`
  unset.
- The run uses a fresh filesystem store root, tenant, output directory, and
  idempotency key. Existing output is not silently overwritten.
- The runner removes and asserts absence of
  `REVIEW_REUSE_DECISIONS_ENABLED`; no decision endpoint is called.
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
  `S3_ENABLED=false`, `EVENT_BUS_ENABLED=false`,
  `ML_PLATFORM_ENABLED=false`, `GEOMETRIC_ENABLED=false`, and `OTEL_ENABLED=false`.
  §5.3 therefore requires visual-only disclosure.
- The current CI candidate is
  `ghcr.io/zensgit/dedupcad-vision@sha256:9f7f567e3b0c1c882f9a363f1b1cb095d30d9e9b184e582d6b19ec7446a86251`.
  This draft does not approve pulling or running it. Any approved pull occurs
  before drawings are mounted or read; the digest is verified before processing.
- The image digest alone is insufficient. The proposed static attestation is
  `docs/development/L3_REVIEW_REUSE_ER3_VISION_SUBSTRATE_ATTESTATION_20260829.json`
  with canonical digest
  `984a49d180dbfa482b5c405a6c8cb2ef7a33078c4dcc88633a566d8b75c0145e`.
  It binds the OCI index to source revision
  `2fc35d60ff034c9f790868c02381a9716becc942`, records both platform manifests and
  expected image IDs, enumerates search-affecting state, and identifies the
  authoritative count and receipt mechanism. Its status is intentionally
  `static_verified_runtime_unverified`.
- Static source/OCI verification and its explicit runtime gaps are recorded in
  `docs/development/L3_REVIEW_REUSE_ER3_SUBSTRATE_STATIC_VERIFICATION_20260829.md`.
- Static attestation is sufficient only for the first owner gate: writing the
  fail-first implementation and mock-backed tests. It does not prove tmpfs,
  empty runtime state, network isolation, selected platform, or cleanup. A second
  explicit owner response at the implementation exact head is required before
  image pull/start or fixture-byte processing. The resulting runtime preflight
  receipt must be captured after container start and before any fixture is read.
- The attestation schema is
  `review-reuse-er3-vision-substrate-attestation-v1` and contains at minimum:
  image reference/digest/expected platform IDs; source repository revision;
  declared and discovered
  mutable database/index/cache paths; the env/config binding and `tmpfs` target
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
- The real-run control transport is fixed-argv `docker exec`/`docker cp` through
  the local Docker daemon. Docker-daemon access is the administrative capability
  boundary. The runner never uses `shell=True`, never accepts a container name,
  path, URL, or command fragment from the manifest, and validates every copied
  destination against the manifest's unique `runtime_name`. Mock HTTP transports
  remain test-only and use `trust_env=False` with no retries.
- The runtime preflight verifies the attested paths against image/runtime
  inspection; maps `/app/data`, `/app/indexes`, `/app/logs`, and `/tmp` to
  run-scoped `tmpfs`; runs the container filesystem read-only; mounts no host data
  path; labels the container with the run ID; and records `docker inspect`
  evidence. Explicit tmpfs overrides are required for all three image-declared
  volumes so Docker cannot create anonymous data volumes. An undeclared mutable
  search path is a hard failure. `docker inspect` must report network mode `none`,
  no additional network attachment, and no published TCP or Unix socket.
- Before any fixture read, fixed argv first requires `curl --version` to exit 0.
  It then invokes the built-in curl as `curl --noproxy '*' --silent --show-error
  --output /dev/null --write-out '%{http_code}' --connect-timeout 2 --max-time 3
  http://198.51.100.1:80/`. The probe must exit with curl network error 7 or 28
  and stdout exactly `000`; exit 127 or any HTTP response is a hard failure.
  `198.51.100.1` is the fixed TEST-NET-2 numeric probe target; DNS and a second
  probe image are not used.
- `GET /health` through `docker exec` to in-container loopback must succeed before
  index mutation and its response is recorded after secret-safe field filtering.
- Before index-add, `GET /api/stats` at JSON pointer
  `/stats/total_drawings` must report zero. The runner also searches the approved
  query and requires zero candidates, but that query is supplementary evidence
  and never substitutes for the authoritative count proof.
- The service caches search responses for five minutes using file MD5 plus mode,
  result limit, diff, ML, and geometry options. The pre-index probe and post-index
  product query use the exact tuples `(fast, 1, false, false, false)` and
  `(balanced, 5, false, false, true)` respectively, ordered as mode, max results,
  compute diff, ML, and geometry. The runner records both tuples and asserts their
  derived cache keys differ. Reusing the same tuple would permit a stale
  zero-result cache hit and is a hard failure.
- After preflight only, each approved archive/query entry is copied to its unique
  `/tmp/<runtime_name>` and removed immediately after its single request. Every
  archive entry is sent to `POST /api/index/add?upload_to_s3=false` by fixed-argv
  in-container curl. A missing or failed receipt fails the run. The query entry is
  never sent to index-add.
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
  The query is sent exactly once on that product path with geometric verification
  requested; the pre-index emptiness probe is recorded separately. Drawing-byte
  index/search commands execute exactly once; command call counts are evidence.
- A service-level `success=true` is insufficient. Every returned candidate must
  match one distinct `(drawing_id, file_hash)` pair from the three receipt set;
  unknown or duplicate identities fail. The result must contain the exact pair
  bound to `archive-exact-001`, and its file hash must equal the query hash. A
  zero-hit response, including `success=true` with empty buckets, fails with
  `archive_recall_incomplete`. This is required because the fixed engine can mask
  an L1 failure as a successful empty response.
- Health, index, rebuild, or search failure produces a non-zero exit and a
  structured failure record. It must not produce a success bundle through the
  ordinary offline `insufficient_evidence` fallback.
- No hosted LLM, external object store, training API, or other egress is allowed.
- The dedicated service is stopped with `docker rm -fv`, then every run-labeled
  container, volume, and network is proven absent. No run network is expected;
  its presence is itself a failure. Cleanup failure is recorded
  and prevents a clean ER3 closeout claim.

### 5.3 Score and verification semantics

- For this fixed service contract, `visual` is exactly the finite numeric
  `levels.l2.feature_similarity` in `[0,1]`, with normalization identifier
  `dedupcad-vision-l2-cosine-v1`. An absent, non-finite, or out-of-range value is
  not clamped or replaced; it fails score-source validation for that candidate.
- `levels.l1.phash_distance` and `levels.l1.similarity` are retained as raw visual
  method evidence, not copied into `visual` and not averaged with L2.
- `semantic` is exactly `levels.l3.semantic_similarity` only when L3 is evidenced.
  It is `null` in the approved run because ML is disabled.
- `geometric` is exactly `levels.l4.geometric_similarity` only when L4 is
  evidenced. It is `null` in the approved run because geometry is disabled.
- Top-level provider fields `similarity` and `confidence` are supplier aggregates.
  They remain in the redacted raw response receipt but are not mapped to any
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
  `ER3ArchiveTaskInput`. That object accepts only the manifest-bound query PNG,
  mapped receipt-bound candidates, and the persisted v2 evidence context. It
  rejects seed candidates and never renames PNG bytes to `.dxf`.
- `create_er3_archive_task()` and public `create_task()` delegate task/event/CAS
  mechanics to one private implementation; the ER3 seam does not fork the legal
  lifecycle. It does not call the legacy recall mapper or ordinary offline
  fallback and cannot turn a failed archive run into `insufficient_evidence`.
- `ReviewReuseTask.evidence_pack_schema_version` is a required-in-memory
  `Literal["evidence-pack-v1", "evidence-pack-v2"]` field. A persisted legacy
  record that lacks it loads as v1. Unknown values fail closed.
- `ReviewReuseTask.evidence_context` is a strict optional
  `ReviewReuseEvidenceContextV2` model. Its exact fields are frozen by the
  contract below. V1 requires it to be null; v2 requires it before the first
  persisted snapshot. Candidate receipt/search provenance remains in each strict
  candidate while the context carries the complete run-level digest bindings.
- The selector is set before the revision-1 pending/running task is first
  persisted and cannot change for the task lifetime. Store create, put, CAS,
  recovery, cancellation, decision reconstruction, and load validation all use
  the task-level selector; they never infer it from a missing or newer pack.
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
- A v2 keyed create uses exactly that object plus
  `"evidence_pack_schema_version": "evidence-pack-v2"`. Reusing a key across
  v1/v2 conflicts. Unkeyed creates retain null idempotency metadata.
- Decision idempotency preimages do not add a redundant selector because they
  already bind `evidence_pack_sha256`; decision reconstruction retains the task
  selector after temporarily removing the reviewed pack.
- The v1 builder and mapper remain frozen. A new v2-only recall mapper and pack
  builder are selected from the immutable task field; shared v1 score coercion is
  not corrected in place.
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
`0045bf4655558ea3ff1342706bd1af1c2eca26304821ebc19c0fe2ac7ff446db`.
It freezes exact object keys, types/nullability, list ordering, task context,
unknown-field rejection, and digest exclusions. Its embedded golden vector has
EvidencePack digest
`3aa3ce7fc94abb80e0693bf2229d9f8c244482cc3eaa488ed80cb173462a33c4`.
The contract digest excludes only `contract_sha256`; the golden EvidencePack
digest excludes only `evidence_pack_sha256`. Implementation copies neither value
from prose: tests recompute both with `canonical_json_v1()` and reject any field,
type, ordering, nullability, or digest drift.

This is a narrowly scoped L3 compatibility change. Owner ratification must
explicitly approve it; otherwise ER3 implementation may not start. Even after
implementation ratification, image runtime remains blocked until the separate
repository-fixture-run owner gate in §10.

### 5.5 Provenance

The strict persisted `ReviewReuseEvidenceContextV2`, candidate provenance, and
success bundle together bind actual observed values or canonical digests for:

- source query SHA-256;
- archive manifest SHA-256;
- successful index receipt set digest;
- search response digest;
- service identity/version payload digest;
- model identifier/version/digest when the service exposes it;
- ReviewReuse score-mapping ruleset digest;
- calibration status and version/digest, with null for unavailable fields;
- exact repository commit and runner version;
- task, trace, idempotency, task revision, and EvidencePack identifiers.

Placeholder values do not satisfy this contract. If a value required to support
the claim is unavailable, the field remains null with a structured reason. A run
that cannot establish archive/index/search provenance cannot close ER3. Every
field needed to rebuild the v2 EvidencePack after process restart lives in the
task or its candidates; exported receipts are evidence bound by digest, not an
unstated reconstruction dependency.

### 5.6 Export and replay

A success run writes the filesystem store, manifest copy, runtime preflight,
redacted index/rebuild/search/service receipts, task JSON, EvidencePack JSON,
EvidencePack Markdown, audit bundle, and run summary. The summary has an exact
artifact-name-to-SHA-256 map for every artifact except itself and records command
posture without secrets. The success stdout records the summary SHA-256; replay
requires it as `--summary-sha256` rather than trusting the file that names its own
children.

Replay has one ordered fact chain:

1. strictly parse the summary and match the required externally supplied summary
   SHA-256;
2. hash every named artifact before opening the store and reject missing, extra,
   renamed, or mismatched artifacts;
3. open the run filesystem store in explicit read-only mode and load the exact
   `(tenant_id, task_id)` through ordinary selector-dispatched store validation;
4. require canonical equality among the loaded task, exported task JSON, the
   task-embedded pack, and independent EvidencePack JSON;
5. validate the pack and contract/golden rules, then regenerate only Markdown and
   audit rendering and require byte equality with their recorded artifacts;
6. hash the store tree before and after replay and require no write, event,
   revision, or file change.

Replay never rebuilds the canonical pack JSON, calls Docker/index/search, or uses
an exported EvidencePack as the primary fact source. Each artifact class and the
summary hash have independent tamper tests. Any mismatch fails before emitting a
success report.

## 6. Failure taxonomy

The CLI exits non-zero with a structured `status="failed"` and one stable
`reason_code` from this minimum vocabulary:

- `manifest_invalid`
- `manifest_content_drift`
- `archive_input_media_invalid`
- `archive_control_transport_invalid`
- `archive_substrate_unattested`
- `archive_runtime_preflight_failed`
- `archive_instance_not_isolated`
- `archive_index_cardinality_unavailable`
- `archive_index_readiness_failed`
- `archive_search_cache_unsafe`
- `archive_recall_incomplete`
- `vision_health_unavailable`
- `archive_index_failed`
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
baseline log showing they fail against the ratified implementation base:

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
13. `test_er3_numeric_bounded_egress_probe_requires_curl_and_network_error`
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

Tests may use a deterministic fake private vision server for failure and mapping
cases. ER3 closure additionally requires a separately owner-authorized recorded
run against the dedicated pinned `dedupcad-vision` container and the approved
manifest. Mock-only green tests are not closure evidence.

Renaming, weakening, deleting, skipping, or marking these tests xfail requires
owner review. Additional narrower tests are allowed.

## 8. Proposed implementation write set

Only after ratification:

- `scripts/review_reuse_isolated_archive_run.py`
- new `src/core/review_reuse/er3_archive.py`
- new `src/core/review_reuse/evidence_v2.py`
- new `src/core/review_reuse/evidence_dispatch.py`
- `src/core/review_reuse/service.py` and `store.py` only for the explicit v1/v2
  selector, internal manifest-bound PNG seam, read-only replay, dispatch,
  context validation, and idempotency rules in §5.4-§5.6
- `src/core/review_reuse/models.py` only for the immutable selector, strict v2
  evidence context, and selector-conditioned calibration described in §5.3-§5.4
- `src/api/v1/review_reuse.py` only to add strict task/EvidencePack response
  models and the selector to `TaskSummary`; public create and decision request
  models and decision semantics remain unchanged
- `config/openapi_schema_snapshot.json` for the newly documented task component;
  the named property-level test, not the snapshot alone, proves selector exposure
- `tests/unit/test_review_reuse_er3_archive.py`
- the exact §4 manifest under `tests/fixtures/review_reuse_er3/`
- one ER3 development/verification document

The static substrate attestation and EvidencePack v2 contract are inputs to
implementation and remain byte-identical. Updating either, changing the fixture
manifest object from §4, or changing the pinned image requires a new design-lock
head and owner response.

The existing v1 files `dedup_live.py`, `dedup_adapter.py`, `evidence.py`, and
`canonical.py` remain byte-identical. If dispatch cannot be added without editing
one of them, implementation stops for a new owner review instead of widening the
write set.

Any change outside this list, any auth/persistence/decision semantic change
beyond the explicit §5.4 version dispatch, or any attempt to combine ER4 requires
a new owner decision. No implementation PR may modify `eval_integrity_gate`,
training/feedback paths, assistant paths, cost-cap code, deployment, or PLM
write-back.

## 9. Acceptance and non-claims

ER3 is complete only when all of the following are true at one exact head:

- the named fail-first baseline is attached and every test is green;
- the owner-ratified static substrate attestation proves the exact image/source,
  mutable-path, configuration, and cardinality mechanisms, and the separately
  authorized runtime preflight and run receipts prove the observed zero-to-three
  transition;
- the runtime manifest matches §4 digest
  `7fd1e774429fa5f75ab5728ed3e2a55b1972d18d8629da584bcf37dc1de91acf`;
- the v2 contract and golden digests remain
  `0045bf4655558ea3ff1342706bd1af1c2eca26304821ebc19c0fe2ac7ff446db`
  and `3aa3ce7fc94abb80e0693bf2229d9f8c244482cc3eaa488ed80cb173462a33c4`;
- existing ReviewReuse, identity, production-preflight, and core-fast suites are
  green without skips added for this tranche;
- an approved real private service run indexes the archive and searches the
  separate query without seeds, host publication, or an unknown candidate, and
  returns the exact receipt-bound archive candidate;
- score dimensions, missing evidence, confidence, and provenance satisfy §5;
- recorded export replay validates the summary and every artifact against the
  read-only persisted store without Docker/network access or any write;
- the development/verification document records exact SHA, commands, results,
  artifact digests, control-transport posture, and remaining limits;
- exact-head CI and independent review have no unresolved failing finding.

Even after those gates pass, the result proves only the controlled ER3 engine
path. It does not prove model quality, customer utility, production readiness,
decision authorization, deployment readiness, or pilot acceptance.

`evidence-pack-v2` remains an internal ER3-runner selection. This document does
not make it the default public create behavior and does not enable legacy v1 live
dedup. Any v2 default migration or v1 retirement is a separate owner decision.

## 10. Ratification texts

Suggested implementation-only owner response:

> I ratify `L3_REVIEW_REUSE_ER3_DESIGNLOCK_20260829.md` at exact head `<sha>`.
> I authorize only ER3 implementation on exact base `<base-sha>`, using the exact
> repository fixture manifest digest
> `7fd1e774429fa5f75ab5728ed3e2a55b1972d18d8629da584bcf37dc1de91acf`, vision image
> `ghcr.io/zensgit/dedupcad-vision@sha256:9f7f567e3b0c1c882f9a363f1b1cb095d30d9e9b184e582d6b19ec7446a86251`, static substrate attestation digest
> `984a49d180dbfa482b5c405a6c8cb2ef7a33078c4dcc88633a566d8b75c0145e`, EvidencePack v2 contract digest
> `0045bf4655558ea3ff1342706bd1af1c2eca26304821ebc19c0fe2ac7ff446db`, golden vector digest
> `3aa3ce7fc94abb80e0693bf2229d9f8c244482cc3eaa488ed80cb173462a33c4`, the versioned
> compatibility contract in §5.4, and the named fail-first contract
> in §7. I do not authorize image pull/start, fixture processing, merge, decision
> enablement, deployment, pilot, or customer data.

Suggested repository-fixture-run owner response after implementation review:

> I authorize one ER3 repository-fixture run at implementation exact head `<sha>`
> using the ratified manifest, image, static attestation, and reviewed command
> `<command-or-runbook-digest>`. The runner must stop before fixture read unless
> runtime preflight satisfies §5.2. I do not authorize merge, decision enablement,
> deployment, pilot, customer data, or any other image/service.

Until the first response names the exact document head, base, fixture manifest,
vision image digest, static attestation digest, EvidencePack v2 contract digest,
golden vector digest, and fail-first contract, this
document remains a proposal and implementation must not start. Until the second
response names the implementation exact head and reviewed command, the image must
not be pulled or started and fixture bytes must not be processed.
