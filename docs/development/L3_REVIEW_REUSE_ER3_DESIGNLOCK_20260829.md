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

Owner ratification must name all four of the following:

1. the exact implementation base SHA;
2. the exact repository-fixture manifest from §4;
3. the digest-pinned private vision execution substrate and attestation digest
   from §5.2;
4. the fail-first contract in §7.

Without all four, ER3 remains blocked. Ratification of this document would
authorize only ER3 implementation and verification. It would not authorize
merging any PR, setting `REVIEW_REUSE_DECISIONS_ENABLED`, deployment, pilot use,
or processing customer data.

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

Use only repository-tracked DXF fixtures under
`tests/fixtures/ci/hybrid_blind_dxf/`. The current tree contains 20 tracked DXF
paths, 15 unique SHA-256 values, and five byte-identical pairs. A minimal manifest
should select one byte-identical pair as separate archive/query roles and at least
two unrelated archive drawings.

Recommended deterministic selection:

| Role | Path relative to fixture root | SHA-256 |
|---|---|---|
| archive | `BTJ00000000000-00UNKNOWNv1.dxf` | `491068ca6b008409bba8fba69f6ebacba94f0210805f01190c0523c10adb65a5` |
| query | `BTJ00000000000-00UNKNOWNv1_fixture1.dxf` | `491068ca6b008409bba8fba69f6ebacba94f0210805f01190c0523c10adb65a5` |
| archive | `J0000000-02UNKNOWNv1.dxf` | `241dfcb4926e51fca23ebd8211c0bf86e1e99072eed99227af35337a79a485a9` |
| archive | `LTJ000000000-0000UNKNOWNv1.dxf` | `06b7026c5dc7366cedde4a2db48b28e08aabccfb8754318998f95abe8a6451c7` |

The exact manifest proposed for ratification is:

```json
{
  "schema_version": "review-reuse-er3-archive-manifest-v1",
  "archive_id": "rr-er3-repository-fixture-v1",
  "archive_manifest_sha256": "e377cc0ef35c60acaf933487662c7d3d60b31411768c53979e933b96d2c59cd6",
  "source_class": "repository_fixture",
  "retention_class": "test_fixture",
  "customer_data": false,
  "entries": [
    {
      "entry_id": "archive-exact-001",
      "role": "archive",
      "path": "tests/fixtures/ci/hybrid_blind_dxf/BTJ00000000000-00UNKNOWNv1.dxf",
      "media_type": "image/vnd.dxf",
      "size_bytes": 55122,
      "sha256": "491068ca6b008409bba8fba69f6ebacba94f0210805f01190c0523c10adb65a5"
    },
    {
      "entry_id": "archive-control-001",
      "role": "archive",
      "path": "tests/fixtures/ci/hybrid_blind_dxf/J0000000-02UNKNOWNv1.dxf",
      "media_type": "image/vnd.dxf",
      "size_bytes": 55122,
      "sha256": "241dfcb4926e51fca23ebd8211c0bf86e1e99072eed99227af35337a79a485a9"
    },
    {
      "entry_id": "archive-control-002",
      "role": "archive",
      "path": "tests/fixtures/ci/hybrid_blind_dxf/LTJ000000000-0000UNKNOWNv1.dxf",
      "media_type": "image/vnd.dxf",
      "size_bytes": 55082,
      "sha256": "06b7026c5dc7366cedde4a2db48b28e08aabccfb8754318998f95abe8a6451c7"
    },
    {
      "entry_id": "query-exact-001",
      "role": "query",
      "path": "tests/fixtures/ci/hybrid_blind_dxf/BTJ00000000000-00UNKNOWNv1_fixture1.dxf",
      "media_type": "image/vnd.dxf",
      "size_bytes": 55122,
      "sha256": "491068ca6b008409bba8fba69f6ebacba94f0210805f01190c0523c10adb65a5",
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
- repository-relative path, role, media type, byte size, and file SHA-256;
- source class fixed to `repository_fixture`;
- retention class fixed to `test_fixture` and `customer_data=false`;
- an optional expected relationship used only for test verification, never as a
  recall candidate or score input.

`archive_manifest_sha256` is computed over canonical JSON after removing that
field itself. Canonical paths must be unique. Duplicate hashes inside the
archive role fail.
A query/archive hash match is allowed only when explicitly declared as a
byte-identical fixture relationship. The query path is never added to the index.
Path escape, symlink escape, missing files, size/hash drift, unknown fields, or
multiple query entries fail before any network call.

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
- The owner-approved vision image is pinned by digest and the actual image ID is
  recorded. The repository's current CI supplies the image/config candidate,
  not the isolation contract: ER3 additionally requires a run-scoped Docker
  `--internal` network, loopback-only host port binding, explicit ephemeral data
  mounts, and verified cleanup. The service runs with `S3_ENABLED=false`,
  `EVENT_BUS_ENABLED=false`, and `ML_PLATFORM_ENABLED=false`. Geometry may be
  enabled only if its complete local dependencies are available; otherwise §5.3
  requires visual-only disclosure.
- The current CI candidate is
  `ghcr.io/zensgit/dedupcad-vision@sha256:9f7f567e3b0c1c882f9a363f1b1cb095d30d9e9b184e582d6b19ec7446a86251`.
  This draft does not approve pulling or running it. Any approved pull occurs
  before drawings are mounted or read; the digest is verified before processing.
- The image digest alone is insufficient. Before ratification, an owner-reviewed
  substrate attestation must bind the image digest to its source/release revision,
  enumerate every database/index/cache path that can affect search, name the
  environment/configuration that redirects each path to run-scoped `tmpfs`, and
  name an authoritative indexed-drawing count/list/reset mechanism. The
  attestation itself is canonical JSON with a SHA-256 named in the owner response.
- The current CI candidate has no such attestation in this repository. Until one
  is supplied, §5.2 is not satisfied and ER3 implementation remains blocked.
- The attestation schema is
  `review-reuse-er3-vision-substrate-attestation-v1` and includes exactly:
  image reference/digest/ID; source repository revision; declared and discovered
  mutable database/index/cache paths; the env/config binding and `tmpfs` target
  for each path; disabled integration flags; the authoritative pre/post indexed-
  drawing count/list command or endpoint and response schema; expected zero and
  post-index counts; network/read-only/port/cleanup posture; and
  `attestation_sha256`. Its digest excludes only `attestation_sha256` itself.
- The service binds only to a literal loopback address, for example
  `127.0.0.1:58001`. URL credentials, DNS hostnames, redirects, proxy inheritance,
  and non-loopback destinations are rejected. The ER3 HTTP transport uses
  `trust_env=False` so `HTTP_PROXY`/`HTTPS_PROXY` cannot redirect drawing bytes.
- The run verifies the attested paths against image/runtime inspection, maps every
  mutable data path to run-scoped `tmpfs`, runs the container filesystem
  read-only, mounts no host data path, labels the container/network with the run
  ID, and records `docker inspect` evidence. An undeclared mutable search path is
  a hard failure. The internal network must report `Internal=true`; a disposable
  probe on that network must fail outbound access; the vision port binding must
  report host IP `127.0.0.1`.
- `health()` must succeed before index mutation and its response is recorded
  after secret-safe field filtering.
- Before index-add, the attested authoritative mechanism must report zero indexed
  drawings. The runner also searches the approved query and requires zero
  candidates, but that query is supplementary evidence and never substitutes for
  the authoritative count/list proof.
- Every archive entry is sent through `index_add_2d(..., upload_to_s3=False)`.
  A missing or failed receipt fails the run.
- Index rebuild is executed only when the approved service contract requires it;
  the choice and response are recorded.
- After index/rebuild, the authoritative mechanism must report exactly the three
  archive entries in §4 and bind their service-side identifiers/file hashes. A
  count/list mismatch fails before the product query.
- The post-index ReviewReuse search runs only after all archive receipts succeed.
  The query is sent exactly once on that product path with geometric verification
  requested; the pre-index emptiness probe is recorded separately. Drawing-byte
  index/search requests use a no-retry client (`retry_max_attempts=1`); transport
  call counts are evidence. Health-only retries do not carry drawing bytes.
- Health, index, rebuild, or search failure produces a non-zero exit and a
  structured failure record. It must not produce a success bundle through the
  ordinary offline `insufficient_evidence` fallback.
- No hosted LLM, external object store, training API, or other egress is allowed.
- The dedicated service is stopped with `docker rm -fv`, then every run-labeled
  container, volume, and network is proven absent. Cleanup failure is recorded
  and prevents a clean ER3 closeout claim.

### 5.3 Score and verification semantics

- `visual` contains only an explicitly visual score returned by the service.
- `semantic` contains only an explicitly semantic/text/embedding score. It is
  `null` when the service did not run such a method.
- `geometric` contains only a deterministic geometric/precision result. Generic
  similarity is not a geometric fallback.
- If geometry was requested but not executed or not evidenced, geometric remains
  `null` and the candidate carries `vision_only_unverified`.
- Missing dimensions remain JSON `null`, never `0`, copied values, inferred
  values, or fabricated defaults.
- Verification methods list only methods evidenced by the response. A method
  name is not inferred from a result bucket.
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

- `ReviewReuseTask.evidence_pack_schema_version` is a required-in-memory
  `Literal["evidence-pack-v1", "evidence-pack-v2"]` field. A persisted legacy
  record that lacks it loads as v1. Unknown values fail closed.
- The selector is set before the revision-1 pending/running task is first
  persisted and cannot change for the task lifetime. Store create, put, CAS,
  recovery, cancellation, decision reconstruction, and load validation all use
  the task-level selector; they never infer it from a missing or newer pack.
- Existing and default API tasks remain v1 and reproduce the exact current raw
  hit mapping, canonical pack bytes, and digest. Only the internal ER3 runner
  explicitly requests v2. The additive task field is exposed honestly in task
  responses: create/get/cancel/decision use the existing `ReviewReuseTask` model
  instead of `Dict[str, Any]`, and list summary includes the selector. This
  requires an updated OpenAPI snapshot plus a property-level contract test, but
  adds no public request switch and does not alter decision request fields.
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

This is a narrowly scoped L3 compatibility change. Owner ratification must
explicitly approve it; otherwise ER3 cannot truthfully close and no runtime work
may start.

### 5.5 Provenance

The success bundle must bind actual observed values or canonical digests for:

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
that cannot establish archive/index/search provenance cannot close ER3.

### 5.6 Export and replay

A success run writes a manifest copy, redacted service receipts, task JSON,
EvidencePack JSON, EvidencePack Markdown, audit bundle, and run summary. The
summary records each artifact SHA-256 and the command posture without secrets.

Replay loads the persisted task, re-emits and digest-verifies the immutable
persisted EvidencePack JSON, and regenerates only Markdown and audit rendering.
It does not rebuild canonical pack JSON and does not call the index or search
service. EvidencePack digest, candidate identifiers/scores/reasons/provenance,
and security posture must match the recorded run. Timestamps already bound to
the stored task are preserved; replay does not create new events.

## 6. Failure taxonomy

The CLI exits non-zero with a structured `status="failed"` and one stable
`reason_code` from this minimum vocabulary:

- `manifest_invalid`
- `manifest_content_drift`
- `endpoint_not_private`
- `archive_substrate_unattested`
- `archive_instance_not_isolated`
- `archive_index_cardinality_unavailable`
- `vision_health_unavailable`
- `archive_index_failed`
- `archive_rebuild_failed`
- `archive_search_failed`
- `archive_provenance_unavailable`
- `score_source_invalid`
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
3. `test_er3_manifest_rejects_path_escape_hash_drift_and_duplicate_roles`
4. `test_er3_query_is_not_added_to_archive_index`
5. `test_er3_real_mode_rejects_seed_candidates`
6. `test_er3_endpoint_must_be_private_and_credential_free`
7. `test_er3_requires_attested_index_roots_and_cardinality_contract`
8. `test_er3_requires_fresh_digest_pinned_ephemeral_vision_instance`
9. `test_er3_index_count_is_zero_then_matches_archive_entry_count`
10. `test_er3_fresh_instance_has_zero_preindex_query_results`
11. `test_er3_http_transport_ignores_proxy_environment`
12. `test_er3_drawing_requests_are_not_retried`
13. `test_er3_index_add_disables_object_store_upload`
14. `test_er3_health_index_and_search_fail_closed`
15. `test_er3_visual_score_is_not_copied_to_semantic_or_geometric`
16. `test_er3_missing_geometry_is_null_and_marks_vision_only_unverified`
17. `test_er3_uncalibrated_run_does_not_emit_confidence`
18. `test_er3_provenance_rejects_placeholders_and_binds_observed_digests`
19. `test_er3_task_selector_is_immutable_and_legacy_defaults_to_v1`
20. `test_er3_v1_raw_mapping_pack_and_decision_digests_are_unchanged`
21. `test_er3_v1_and_v2_create_digest_preimages_are_exact`
22. `test_er3_store_dispatches_v1_and_v2_without_silent_upgrade`
23. `test_er3_cancel_and_decision_reconstruction_retain_task_selector`
24. `test_er3_unknown_evidence_pack_version_fails_closed`
25. `test_er3_openapi_exposes_selector_without_public_create_switch`
26. `test_er3_replay_reemits_persisted_pack_without_network_calls`
27. `test_er3_json_markdown_and_audit_export_are_consistent`
28. `test_er3_runner_never_enables_or_submits_decisions`
29. `test_er3_container_is_loopback_internal_network_and_no_egress`
30. `test_er3_dedicated_instance_and_volumes_are_cleaned_up`

Tests may use a deterministic fake private vision server for failure and mapping
cases. ER3 closure additionally requires one recorded run against the real local
`dedupcad-vision` process and the approved manifest. Mock-only green tests are not
closure evidence.

Renaming, weakening, deleting, skipping, or marking these tests xfail requires
owner review. Additional narrower tests are allowed.

## 8. Proposed implementation write set

Only after ratification:

- `scripts/review_reuse_isolated_archive_run.py`
- new `src/core/review_reuse/er3_archive.py`
- new `src/core/review_reuse/evidence_v2.py`
- new `src/core/review_reuse/evidence_dispatch.py`
- `src/core/dedupcad_vision.py` only to permit an explicit proxy-free transport
  without changing the default posture of other callers
- `src/core/review_reuse/service.py` and `store.py` only for the explicit v1/v2
  selector, dispatch, and idempotency rules in §5.4
- `src/core/review_reuse/models.py` only for the immutable selector and nullable
  v2 calibration version described in §5.3-§5.4
- `src/api/v1/review_reuse.py` only to replace generic task response models with
  `ReviewReuseTask` and add the selector to `TaskSummary`; request models and
  decision semantics remain unchanged
- `config/openapi_schema_snapshot.json` for the newly documented task component;
  the named property-level test, not the snapshot alone, proves selector exposure
- `tests/unit/test_review_reuse_er3_archive.py`
- the exact §4 manifest under `tests/fixtures/review_reuse_er3/`
- one ER3 development/verification document

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
- the owner-ratified substrate attestation proves authoritative index roots and
  zero-to-three indexed-drawing cardinality for the exact image digest;
- the runtime manifest matches §4 digest
  `e377cc0ef35c60acaf933487662c7d3d60b31411768c53979e933b96d2c59cd6`;
- existing ReviewReuse, identity, production-preflight, and core-fast suites are
  green without skips added for this tranche;
- an approved real private service run indexes the archive and searches the
  separate query without seeds;
- score dimensions, missing evidence, confidence, and provenance satisfy §5;
- recorded export replay passes without network access;
- the development/verification document records exact SHA, commands, results,
  artifact digests, endpoint posture, and remaining limits;
- exact-head CI and independent review have no unresolved failing finding.

Even after those gates pass, the result proves only the controlled ER3 engine
path. It does not prove model quality, customer utility, production readiness,
decision authorization, deployment readiness, or pilot acceptance.

`evidence-pack-v2` remains an internal ER3-runner selection. This document does
not make it the default public create behavior and does not enable legacy v1 live
dedup. Any v2 default migration or v1 retirement is a separate owner decision.

## 10. Ratification text

Suggested owner response:

> I ratify `L3_REVIEW_REUSE_ER3_DESIGNLOCK_20260829.md` at exact head `<sha>`.
> I authorize only ER3 implementation on exact base `<base-sha>`, using the exact
> repository fixture manifest digest
> `e377cc0ef35c60acaf933487662c7d3d60b31411768c53979e933b96d2c59cd6`, vision image
> `<image@sha256:...>`, substrate attestation digest `<sha256>`, the versioned
> EvidencePack compatibility contract in §5.4, and the named fail-first contract
> in §7. I do not authorize merge, decision enablement, deployment, pilot, or
> customer data.

Until that response names the exact document head, base, fixture manifest, vision
image digest, and substrate attestation digest, this document remains a proposal
and runtime work must not start.
