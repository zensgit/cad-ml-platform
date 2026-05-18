# CAD ML Reviewer Template Preflight Manifest Match Development

Date: 2026-05-14

## Goal

Move reviewer-template row identity failures from apply time into preflight.
Returned reviewer templates can now be checked against the current review
manifest before they are applied, so a row with a stale or edited identity is
blocked early and appears in the preflight report and gap CSV.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - `validate_reviewer_template_rows` accepts optional `base_rows`.
  - Preflight summaries now include:
    - `base_manifest_match_required`
    - `base_manifest_row_count`
    - `unmatched_template_row_count`
  - Ready row counting now requires a manifest match when a base manifest is
    supplied.
  - Preflight Markdown reports unmatched manifest rows.
  - Preflight gap CSV includes `matched_manifest_row`.
  - `--validate-reviewer-template` now honors optional `--base-manifest`.
  - Generated handoff preflight commands include `--base-manifest` when the
    review manifest path is known.
- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Reviewer-template preflight now passes the current review manifest as
    `--base-manifest` before apply.
- Updated tests.
  - Covers blocked unmatched template rows.
  - Covers no-base default behavior as `not_checked`.
  - Covers CLI preflight with a matching base manifest.
  - Covers wrapper-level blocking before apply.
- Updated Phase 6 TODO.

## Claude Code Review

Claude Code is available locally as `/Users/chouhua/.local/bin/claude` and was
used as a read-only reviewer for this slice. Tools were disabled and only
targeted code/test snippets were sent.

Claude found no default-path regression. It specifically called out that the
legacy no-base path remains unchanged, the wrapper passes the current review
manifest, and the schema additions are additive. Its follow-up checks were
covered by local assertions for `not_checked` no-base rows and handoff command
parity.

## Release Impact

This does not approve labels or change release thresholds. It makes the manual
review loop stricter and easier to debug: a filled batch template must both pass
label/metadata checks and match the current review manifest before apply.

## Remaining Work

- Have qualified reviewers fill real source, payload, and `details.*` labels.
- Preflight returned templates against the active review manifest.
- Apply only preflight-ready templates.
- Tune thresholds after the reviewed release benchmark set is stable.
