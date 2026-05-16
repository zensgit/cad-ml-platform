# CAD ML Reviewed Manifest Hybrid Gate Development

Date: 2026-05-13

## Goal

Make release-labelled hybrid blind evaluation fail fast when it does not consume the
merged reviewed benchmark manifest. The workflow already prefers the reviewed
manifest when available; this slice adds an optional gate so release jobs can require
that behavior instead of relying on manual log inspection.

## Changes

- Updated `scripts/ci/check_hybrid_blind_gate.py`.
  - Added `manifest_source` to the gate input summary.
  - Added `required_manifest_source` to the gate input summary.
  - Added `--manifest-source`.
  - Added `--require-manifest-source`.
  - Fails the gate when the actual manifest source does not match the required
    manifest source.
- Updated `.github/workflows/evaluation-report.yml`.
  - Added repository variable wiring:
    `HYBRID_BLIND_REQUIRE_REVIEWED_MANIFEST`.
  - The hybrid blind gate now passes
    `steps.hybrid_blind_eval.outputs.manifest_source` into
    `scripts/ci/check_hybrid_blind_gate.py`.
  - When `HYBRID_BLIND_REQUIRE_REVIEWED_MANIFEST=true`, the gate requires:
    `manifest_source=reviewed_benchmark_manifest`.
- Updated tests.
  - Unit coverage verifies `configured` manifest sources fail when
    `reviewed_benchmark_manifest` is required.
  - Workflow regression coverage verifies the new env var and gate CLI wiring.
- Updated Phase 6 TODO.

## Configuration

Default non-blocking mode:

```bash
HYBRID_BLIND_REQUIRE_REVIEWED_MANIFEST=false
```

Release-labelled reviewed-manifest mode:

```bash
HYBRID_BLIND_REQUIRE_REVIEWED_MANIFEST=true
```

In release mode, the hybrid blind gate passes only when the hybrid benchmark reports:

```text
manifest_source=reviewed_benchmark_manifest
```

## Release Impact

This closes the verification gap after reviewed-manifest merge wiring. Once domain
reviewers approve the release labels, CI can enforce that hybrid blind evaluation uses
the merged reviewed benchmark manifest rather than a manually configured or stale
manifest path.

## Remaining Work

- Populate the real release benchmark review manifest with domain-approved source,
  payload, and detail labels.
- Run a release-labelled evaluation with
  `HYBRID_BLIND_REQUIRE_REVIEWED_MANIFEST=true`.
- Tune source, payload, and detail quality thresholds after the reviewed set is
  stable.
