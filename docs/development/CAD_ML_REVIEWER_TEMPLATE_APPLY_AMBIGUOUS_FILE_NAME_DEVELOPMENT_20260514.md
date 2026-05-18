# CAD ML Reviewer Template Apply Ambiguous File Name Development

Date: 2026-05-14

## Goal

Close the remaining direct reviewer-template apply ambiguity where a returned
template row can omit or mistype `relative_path`, match a duplicate `file_name`
in the base review manifest, and silently update the first matching row.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - `apply_reviewer_template_rows` now detects duplicate base review manifest
    file names.
  - Direct apply blocks file-name fallback when the base review manifest has
    the same `file_name` under multiple `relative_path` values and the template
    row does not provide a precise matching `relative_path`.
  - Apply summaries now include `ambiguous_file_name_match_row_count`.
  - Apply summaries add `ambiguous_file_name_match_rows` to blocking reasons.
  - Apply status is blocked whenever blocking reasons exist, including mixed
    batches where precise rows apply but ambiguous fallback rows are skipped.
  - Apply audit rows mark ambiguous fallback rows as
    `ambiguous_file_name_match`.
- Updated tests.
  - Covers missing `relative_path` with duplicate base file names.
  - Covers mistyped `relative_path` with duplicate base file names.
  - Covers precise `relative_path` matches continuing to apply normally.
  - Covers mixed batches where precise rows apply but ambiguous rows keep the
    summary blocked.
  - Covers apply audit output for ambiguous file-name fallback rows.
- Updated Phase 6 TODO.

## Claude Code Review

Claude Code is available locally as `/Users/chouhua/.local/bin/claude` and was
used as a read-only reviewer for targeted snippets. Tools were disabled and no
secrets or environment values were sent.

Claude Code found no regression in the focused apply path. It confirmed precise
`relative_path` matches are preserved, ambiguous file-name fallback is blocked,
mixed batches keep successful precise writes while returning blocked status, and
apply audit status mirrors the apply summary behavior. It also called out the
mistyped `relative_path` case; this slice added a regression assertion for that
case.

## Release Impact

No labels are approved and no release thresholds are changed. This makes direct
reviewer-template apply safer when it is run outside the generated
preflight/handoff workflow or when a returned template has incomplete row
identity fields.

## Remaining Work

- Populate real reviewed source, payload, and `details.*` labels.
- Keep reviewer templates preflighted against the active, non-ambiguous review
  manifest before apply.
- Apply only preflight-ready reviewer outputs.
- Tune thresholds after the reviewed release benchmark set is stable.
