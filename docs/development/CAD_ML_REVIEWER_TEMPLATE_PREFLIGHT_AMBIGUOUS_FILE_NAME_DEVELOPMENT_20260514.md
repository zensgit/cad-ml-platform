# CAD ML Reviewer Template Preflight Ambiguous File Name Development

Date: 2026-05-14

## Goal

Move duplicate `file_name` ambiguity detection earlier in the reviewer-template
workflow. A returned template row that omits or mistypes `relative_path` should
fail preflight when the base review manifest has the same `file_name` under
multiple paths, instead of looking ready and failing later during apply.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - `validate_reviewer_template_rows` now detects duplicate base manifest file
    names.
  - Preflight blocks file-name fallback when the template row does not provide a
    precise matching `relative_path`.
  - Preflight summaries now include `ambiguous_file_name_match_row_count`.
  - Preflight blocking reasons now include `ambiguous_file_name_match_rows`.
  - Ambiguous rows are not double-counted as unmatched manifest rows.
  - Duplicate base row identities keep precedence over file-name ambiguity.
  - Preflight Markdown now includes an ambiguous fallback count.
  - Preflight gap CSV rows now include `ambiguous_file_name_match`.
- Updated tests.
  - Covers missing `relative_path` with duplicate base file names.
  - Covers mistyped `relative_path` with duplicate base file names.
  - Covers precise `relative_path` matches remaining ready.
  - Covers Markdown reporting for ambiguous fallback rows.
  - Covers preflight gap CSV diagnostics for ambiguous fallback rows.
- Updated Phase 6 TODO.

## Claude Code Review

Claude Code is available locally as `/Users/chouhua/.local/bin/claude` and was
used as a read-only reviewer for targeted snippets. Tools were disabled and no
secrets or environment values were sent.

Claude Code found no regression in duplicate-base precedence or counter
accounting. It confirmed ambiguous rows are not double-counted as unmatched rows
and noted a redundant ready-row guard; the redundant guard was removed while
keeping the same behavior.

## Release Impact

No labels are approved and no release thresholds are changed. This makes
preflight stricter and better aligned with direct apply, so incomplete reviewer
templates are corrected before they enter the apply step.

## Remaining Work

- Populate real reviewed source, payload, and `details.*` labels.
- Keep reviewer templates preflighted against the active, non-ambiguous review
  manifest before apply.
- Apply only preflight-ready reviewer outputs.
- Tune thresholds after the reviewed release benchmark set is stable.
