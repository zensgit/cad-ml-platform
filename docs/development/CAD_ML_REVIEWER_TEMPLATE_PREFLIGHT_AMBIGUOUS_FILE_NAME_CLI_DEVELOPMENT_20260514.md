# CAD ML Reviewer Template Preflight Ambiguous File Name CLI Development

Date: 2026-05-14

## Goal

Pin the command-line contract for ambiguous reviewer-template file-name
fallbacks. The function-level preflight already blocks rows that omit
`relative_path` when the base review manifest has duplicate `file_name` values;
the CLI must also preserve that blocker in JSON, Markdown, CSV, and exit-code
behavior.

## Changes

- Updated `tests/unit/test_build_manufacturing_review_manifest.py`.
  - Added an end-to-end `main(...)` regression for
    `--validate-reviewer-template` with a duplicate-file-name base manifest.
  - Verifies `--fail-under-minimum` returns `1` when the ambiguous row blocks
    preflight.
  - Verifies the summary JSON includes:
    - `mode`
    - `reviewer_template`
    - `base_manifest`
    - `template_row_count`
    - `base_manifest_row_count`
    - `ready_template_row_count`
    - `unmatched_template_row_count`
    - `ambiguous_file_name_match_row_count`
    - exact blocking reasons
  - Verifies the preflight Markdown renders the title, ready count, unmatched
    count, ambiguous fallback count, affected row, and fix guidance.
  - Verifies the preflight gap CSV schema includes
    `ambiguous_file_name_match` and preserves an empty `relative_path`.
  - Verifies the same blocked preflight returns `0` when
    `--fail-under-minimum` is omitted, while still writing blocked summary data.
- Updated Phase 6 TODO.

## Claude Code Review

Claude Code is available locally as `/Users/chouhua/.local/bin/claude` and was
used as a read-only reviewer for the CLI implementation and test snippets. Tools
were disabled and no secrets or environment values were sent.

Claude Code confirmed the original test covered summary, Markdown, gap CSV, and
fail-under-minimum behavior. It recommended stronger assertions for summary
echo fields, exact blocking reasons, Markdown structure, CSV schema, empty
`relative_path`, and the no-fail flag path; this slice added those assertions.

## Release Impact

No production behavior or thresholds changed in this slice. It protects the
reviewer-template preflight artifact contract so CI and reviewers see the same
ambiguous-file-name blocker across all generated outputs.

## Remaining Work

- Populate real reviewed source, payload, and `details.*` labels.
- Keep reviewer templates preflighted against the active, non-ambiguous review
  manifest before apply.
- Apply only preflight-ready reviewer outputs.
- Tune thresholds after the reviewed release benchmark set is stable.
