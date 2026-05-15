# CAD ML Review Handoff Partial Preflight Development

Date: 2026-05-14

## Goal

Make generated manufacturing review handoff commands match the partial batch
workflow. Reviewer-template preflight can now use a lower min-ready-row threshold
than release validation, so the handoff Markdown must show that same value
instead of always showing the release minimum.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - Adds `--reviewer-template-preflight-min-ready-rows`.
  - Passes the value through `_write_handoff_markdown`.
  - Renders the preflight command with the partial threshold.
  - Keeps the apply command on the release `--min-reviewed-samples` value.
  - Prefers the generated batch reviewer template in executable handoff
    preflight/apply commands when both batch and full templates are available.
  - Adds `--reviewer-template-preflight-gap-csv` to the handoff preflight
    command.
- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Passes `MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MIN_READY_ROWS` into
    review-manifest validation so generated handoff artifacts match CI behavior.
- Updated tests.
  - Covers explicit partial preflight threshold rendering.
  - Covers default fallback where preflight and apply both use the release
    minimum.
  - Covers batch-template command selection while keeping the full reviewer
    template in the artifact map.
  - Covers wrapper-generated handoff Markdown with distinct partial and release
    thresholds.
- Updated Phase 6 TODO.

## Claude Code Review

Claude Code is available locally as `/Users/chouhua/.local/bin/claude` and was
used as a read-only reviewer for the relevant snippets. Tools were disabled and
only targeted code/test snippets were sent.

Claude's review found no behavior regression. It suggested:

- Add coverage for the default preflight-threshold fallback.
- Assert wrapper-generated handoff Markdown includes distinct partial and release
  thresholds.
- Include the preflight gap CSV argument in the handoff command if the handoff is
  meant to be executable.

Those suggestions were implemented in this slice.

A second read-only Claude Code pass over the final batch-template command
preference found no behavior regression against the intended priority:
batch template, full template, placeholder.

## Release Impact

Reviewer handoff artifacts now align with the partial-batch flow:

- small reviewer batches can be preflighted with a small min-ready-row value;
- handoff commands target the bounded batch template reviewers are expected to
  fill for the current batch;
- returned approved rows can be applied;
- the full release manifest still remains blocked until release source, payload,
  and detail minimums are met.

## Remaining Work

- Use the generated batch template and partial handoff commands with real
  reviewers.
- Apply approved batches until release minimums are reached.
- Tune thresholds only after the reviewed release set is stable.
