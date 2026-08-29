# ReviewReuse ER3 Substrate Static Verification

**Status:** STATIC VERIFIED / RUNTIME UNVERIFIED / DOCS ONLY

**Date:** 2026-08-29

**CAD ML base:** open PR #584 exact head
`13055d4966fb3b543d747759ebd237d24a55e452`

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
| Static attestation digest | `984a49d180dbfa482b5c405a6c8cb2ef7a33078c4dcc88633a566d8b75c0145e` |
| EvidencePack v2 contract digest | `0045bf4655558ea3ff1342706bd1af1c2eca26304821ebc19c0fe2ac7ff446db` |
| Embedded EvidencePack golden digest | `3aa3ce7fc94abb80e0693bf2229d9f8c244482cc3eaa488ed80cb173462a33c4` |
| Vision OCI index | `sha256:9f7f567e3b0c1c882f9a363f1b1cb095d30d9e9b184e582d6b19ec7446a86251` |

The manifest, attestation, contract, and embedded golden digests were recomputed
with `src.core.review_reuse.canonical.canonical_sha256()` after removing only
their respective self-digest field. Strict I-JSON parsing passed for both JSON
artifacts. The golden vector is a serialization and digest vector, not model-
quality evidence.

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
  `PerceptualHash.compute_hash()` and `FeatureExtractor.extract()`.
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
explicit run-scoped tmpfs mounts over a read-only root. Explicit tmpfs overrides
also prevent image-declared anonymous volumes.

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
index result to include the exact `(drawing_id, file_hash)` pair bound by the
`archive-exact-001` receipt; a zero-hit or unknown-candidate response is failure
even when the provider reports `success=true`.

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
git diff --check
```

Results:

- fixture byte sizes and SHA-256 values matched the proposed manifest;
- manifest canonical digest matched;
- static attestation strict parse and canonical digest matched;
- v2 contract strict parse, canonical digest, and embedded golden digest matched;
- source paths and endpoint semantics above matched exact revision `2fc35d60...`;
- the design names 48 unique fail-first tests and preserves public DXF/v1 behavior;
- the existing canonical, EvidencePack golden, and ER1 store-integrity regression
  selection passed `120/120` under Python 3.11;
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
- read-only root/tmpfs/anonymous-volume posture;
- Docker network mode `none`, absence of additional attachments or host-published
  TCP/Unix sockets, or the in-container loopback bind;
- fixed-argv Docker control, curl availability plus the exact numeric bounded
  no-egress probe exit/status contract, or the
  Docker-daemon administrative boundary;
- observed zero/three counts or L1/L2 sizes;
- image startup, PNG decoding, index receipts, rebuild, search, cleanup, or replay.

Two independent read-only Claude Code CLI review attempts (Opus, then Sonnet)
produced no review text within bounded waits and were terminated. They made no
file changes. This is recorded as **independent review not completed**, not a
passing second opinion. Separate bounded read-only Sol and Terra reviews did
complete. Their source-confirmed findings drove the v2 contract, internal PNG
seam, replay fact chain, no-host-publication control, and exact receipt-bound
candidate requirements above. Those reviews are design evidence only, not runtime
verification.

## 8. Static conclusion

The fixed image is a plausible ER3 visual substrate only for manifest-bound PNG
input and only under the corrected isolation, rebuild, cache, exact-candidate,
score-mapping, versioned EvidencePack, and separate implementation/runtime owner
gates. The static attestation and v2 contract are ready for review but do not
satisfy runtime preflight and grant neither ER3 implementation nor run authority.
