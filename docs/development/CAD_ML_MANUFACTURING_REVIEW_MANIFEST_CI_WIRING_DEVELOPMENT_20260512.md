# CAD ML Manufacturing Review Manifest CI Wiring Development

Date: 2026-05-12

## Goal

Wire the manufacturing evidence review manifest validator into the optional forward
scorecard CI/release flow. Release jobs can now publish validation evidence for the
reviewed source, payload, and nested detail labels before claiming manufacturing
evidence readiness.

## Changes

- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Accepts review manifest paths from:
    - `BENCHMARK_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_CSV`
    - workflow input alias `forward_scorecard_manufacturing_review_manifest_csv`
    - `FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_CSV`
  - Runs `scripts/build_manufacturing_review_manifest.py --validate-manifest` when a
    configured manifest exists.
  - Writes validation summary JSON to
    `FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_SUMMARY_JSON`, defaulting to:

```text
reports/benchmark/forward_scorecard/manufacturing_review_manifest_validation.json
```

  - Uses `FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MIN_REVIEWED_SAMPLES`,
    default `30`.
  - Supports
    `FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_REQUIRE_REVIEWER_METADATA=true`
    to require both reviewer identity and review timestamp.
  - Emits GitHub outputs:
    - `manufacturing_review_manifest_available`
    - `manufacturing_review_manifest_csv`
    - `manufacturing_review_manifest_summary_json`
    - `manufacturing_review_manifest_status`
  - Supports `FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_FAIL_ON_BLOCKED=true`
    to fail the optional scorecard job when the reviewed label manifest is below the
    configured sample threshold.
  - Passes the generated validation summary into the forward scorecard exporter so
    the scorecard artifact itself records the review manifest validation evidence.
- Updated `.github/workflows/evaluation-report.yml`.
  - Adds repository-variable wiring for the review manifest path, summary path,
    minimum reviewed sample count, and fail-on-blocked toggle.
  - Uploads the validation summary as
    `manufacturing-evidence-review-manifest-validation-${{ github.run_number }}` when
    the wrapper reports a manifest was validated.
- Updated tests.
  - Wrapper ready path now validates a release-label-ready review manifest.
  - Wrapper blocked path fails when fail-on-blocked is enabled.
  - Workflow tests assert env wiring, artifact upload wiring, and upload order.

## Configuration

Non-blocking evidence publication:

```text
FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_CSV=reports/experiments/<run>/manufacturing_review_manifest.csv
FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_SUMMARY_JSON=reports/benchmark/forward_scorecard/manufacturing_review_manifest_validation.json
FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MIN_REVIEWED_SAMPLES=30
FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_REQUIRE_REVIEWER_METADATA=false
FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_FAIL_ON_BLOCKED=false
```

Release-blocking mode:

```text
FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_FAIL_ON_BLOCKED=true
```

## Release Impact

The forward scorecard now also records the review manifest validation summary under
the manufacturing evidence component and downgrades manufacturing evidence when a
provided validation summary is blocked. This makes the CI-level precondition visible
in the artifact that release claims cite.

## Remaining Work

- Populate the actual release benchmark manifest with human-reviewed manufacturing
  source, payload, and detail labels.
- Turn `FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_FAIL_ON_BLOCKED` on for
  release-labelled jobs after the release manifest path is stable.
- Add reviewer ownership columns if multiple domain reviewers maintain the manifest.
