# CAD ML Manufacturing Review Progress Report Development

Date: 2026-05-13

## Goal

Make the remaining human review work visible and actionable. The review manifest
validator already reports JSON counts; this slice adds a Markdown progress report
that lists missing source, payload, detail, approval, and reviewer metadata work for
the next rows reviewers should close.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - Added `build_review_progress_markdown`.
  - Added `--progress-md`.
  - Added `--max-progress-rows`.
  - Build and validate modes can now emit a Markdown review progress report.
  - The report includes:
    - validation status
    - minimum reviewed sample target
    - source, payload, and detail reviewed counts
    - remaining counts for each evidence class
    - blocking reasons
    - per-label progress table
    - next gap rows with required reviewer actions
  - Gap reasons include:
    - missing reviewed source labels
    - missing reviewed payload labels
    - missing `details.*` payload labels
    - unapproved review status
    - missing reviewer or reviewed timestamp when metadata is required
- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Adds
    `FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_PROGRESS_MD`.
  - Passes `--progress-md` to the review manifest validator.
  - Emits `manufacturing_review_manifest_progress_md` as a GitHub output.
- Updated `.github/workflows/evaluation-report.yml`.
  - Adds repository-variable wiring for the progress Markdown path.
  - Uploads the progress Markdown alongside the review validation summary artifact.
- Updated tests for:
  - Markdown count and gap-row content
  - CLI progress Markdown generation
  - forward scorecard wrapper outputs
  - workflow artifact path wiring
- Updated Phase 6 TODO.

## CLI Usage

```bash
python scripts/build_manufacturing_review_manifest.py \
  --validate-manifest reports/experiments/<run>/manufacturing_review_manifest.csv \
  --summary-json reports/benchmark/forward_scorecard/manufacturing_review_manifest_validation.json \
  --progress-md reports/benchmark/forward_scorecard/manufacturing_review_manifest_progress.md \
  --min-reviewed-samples 30 \
  --require-reviewer-metadata
```

## Release Impact

The release reviewer no longer has to infer remaining work from JSON counters. The
Markdown artifact names specific rows and actions, so the next release closeout can
focus on domain labeling rather than tooling interpretation.

## Remaining Work

- Populate the real release benchmark review manifest with domain-approved source,
  payload, and detail labels.
- Re-run the validator and use the progress Markdown until source, payload, and
  detail counts all meet the release threshold.
- Tune source, payload, and detail quality thresholds after the reviewed set is
  stable.
