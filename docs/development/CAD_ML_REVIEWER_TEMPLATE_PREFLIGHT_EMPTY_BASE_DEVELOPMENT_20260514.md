# CAD ML Reviewer Template Preflight Empty Base Development

Date: 2026-05-14

## Goal

Make reviewer-template preflight failures clearer when the supplied base review
manifest exists but contains no rows. This can happen when a CI artifact is
misconfigured or truncated. The reviewer should see an artifact-level blocker,
not only row-level unmatched diagnostics.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - `validate_reviewer_template_rows` now adds `base_manifest_empty` when
    `--base-manifest` is supplied but has zero rows.
  - Existing unmatched-row checks remain active, so returned template rows still
    show `unmatched_template_rows` and row-level `matched_manifest_row=false`.
- Updated tests.
  - Covers summary blocking for an empty base manifest.
  - Covers preflight Markdown rendering of `base_manifest_empty`.
- Updated Phase 6 TODO.

## Claude Code Review

Claude Code is available locally as `/Users/chouhua/.local/bin/claude` and was
used as a read-only reviewer. Tools were disabled and only targeted snippets
were sent.

Claude previously identified empty base manifests as a practical risk for the
manifest-match preflight flow. This slice implements that guard explicitly and
keeps the default no-base path unchanged.

## Release Impact

No labels are approved and no release thresholds are changed. The impact is
diagnostic: CI and reviewers can distinguish a broken base artifact from an
ordinary returned-row mismatch.

## Remaining Work

- Populate real reviewed source, payload, and `details.*` labels.
- Preflight returned templates against the active review manifest.
- Apply only preflight-ready templates.
- Tune thresholds after the reviewed release benchmark set is stable.
