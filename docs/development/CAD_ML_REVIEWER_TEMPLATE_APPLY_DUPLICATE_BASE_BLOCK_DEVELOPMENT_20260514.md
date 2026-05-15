# CAD ML Reviewer Template Apply Duplicate Base Block Development

Date: 2026-05-14

## Goal

Close the bypass path where a reviewer template could be applied directly
against a base review manifest with duplicate row identities. Preflight already
detects ambiguous base manifests; direct apply now enforces the same safety
rule.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - `apply_reviewer_template_rows` now detects duplicate base row identities.
  - Direct apply skips all writeback when the base review manifest is ambiguous.
  - Apply summaries now include:
    - `base_manifest_duplicate_identity_count`
    - `base_manifest_duplicate_identifiers`
  - Apply summaries add `base_manifest_duplicate_rows` to blocking reasons.
  - Apply audit rows mark otherwise eligible rows as
    `blocked_duplicate_base_manifest`.
- Updated tests.
  - Covers direct apply blocking with duplicate base identities.
  - Covers apply audit output for duplicate base manifests.
- Updated Phase 6 TODO.

## Claude Code Review

Claude Code is available locally as `/Users/chouhua/.local/bin/claude` and was
used as a read-only reviewer for targeted snippets. Tools were disabled and no
secrets or environment values were sent.

Claude Code found no default-path regression and confirmed the duplicate-base
apply block is additive: clean base manifests still apply normally, while
ambiguous base manifests return blocked summaries and audit diagnostics.

## Release Impact

No labels are approved and no release thresholds are changed. This makes apply
safe even when run outside the generated preflight/handoff workflow.

## Remaining Work

- Populate real reviewed source, payload, and `details.*` labels.
- Preflight returned templates against the active, non-ambiguous review manifest.
- Apply only preflight-ready templates.
- Tune thresholds after the reviewed release benchmark set is stable.
