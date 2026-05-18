# CAD ML Manufacturing Review Handoff Development

Date: 2026-05-14

## Goal

Reduce the remaining manual-review bottleneck without fabricating domain labels.
The release path still needs real approved source, payload, and `details.*`
labels, so this slice adds a single Markdown handoff artifact that packages the
review status, required artifacts, and exact preflight/apply commands for
reviewer closeout.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - Adds `build_review_handoff_markdown`.
  - Adds `--handoff-md` for build and validate modes.
  - Includes validation status, row count, required review sample count, reviewer
    metadata mode, remaining source/payload/detail label counts, artifact paths,
    preflight/apply commands, and blocking reasons.
  - Explicitly warns reviewers not to copy suggestions into reviewed fields
    without human review.
- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Adds `FORWARD_SCORECARD_MANUFACTURING_REVIEW_HANDOFF_MD`.
  - Writes the handoff Markdown during review manifest validation.
  - Emits `manufacturing_review_handoff_md` as a GitHub output.
- Updated `.github/workflows/evaluation-report.yml`.
  - Adds repository-variable wiring for the handoff path.
  - Uploads the handoff Markdown with the review manifest validation artifact.
- Updated targeted tests for:
  - handoff Markdown content and commands
  - CLI `--handoff-md` output
  - optional forward scorecard wrapper output wiring
  - workflow env and artifact upload wiring
- Updated Phase 6 TODO.

## Default CI Path

The default output path is:

```text
reports/benchmark/forward_scorecard/manufacturing_review_handoff.md
```

It can be overridden with:

```text
FORWARD_SCORECARD_MANUFACTURING_REVIEW_HANDOFF_MD=<path>
```

## Reviewer Flow

The generated handoff points reviewers to:

- review manifest validation summary
- progress Markdown
- gap CSV
- assignment plan
- reviewer fill template CSV
- reviewer-template preflight Markdown path

It also embeds the local commands for:

- validating a filled reviewer template
- applying a preflight-ready template back into the review manifest

## Release Impact

This does not change release thresholds or automatically approve any labels. It
adds a human-facing closeout page so the remaining release-label work can be
executed and audited from one artifact.

## Remaining Work

- Use the handoff artifact to drive real reviewer labeling.
- Preflight the filled reviewer template and resolve blocking rows.
- Apply approved rows, merge into the benchmark manifest, and re-run the forward
  scorecard.
- Tune thresholds only after the reviewed release set is stable.
