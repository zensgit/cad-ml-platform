# CAD ML Review Manifest Merge Duplicate Base Block Development

Date: 2026-05-14

## Goal

Close the remaining duplicate-base bypass in the approved review-manifest merge
path. If the base benchmark manifest has duplicate row identities, merging
approved manufacturing labels should block instead of silently updating the
first matching base row.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - `merge_approved_review_rows` now detects duplicate base benchmark manifest
    row identities.
  - Merge now blocks row-level file-name fallback when the base benchmark
    manifest has the same file name under multiple relative paths and the review
    row does not provide a matching `relative_path`.
  - Merge skips all writeback when the base benchmark manifest is ambiguous.
  - Merge summaries now include:
    - `base_manifest_duplicate_identity_count`
    - `base_manifest_duplicate_identifiers`
  - Merge summaries add `base_manifest_duplicate_rows` to blocking reasons.
  - Merge summaries add `ambiguous_file_name_match_rows` when file-name fallback
    would be ambiguous.
  - Merge status is blocked whenever blocking reasons exist, including mixed
    batches where some rows merge but other rows are ambiguous.
  - Merge audit rows mark otherwise eligible rows as
    `blocked_duplicate_base_manifest`.
  - Merge audit rows mark ambiguous file-name fallback rows as
    `ambiguous_file_name_match`.
- Updated tests.
  - Covers approved review-manifest merge blocking with duplicate base
    identities.
  - Covers ambiguous file-name fallback blocking while still allowing precise
    `relative_path` matches.
  - Covers mixed batches where precise rows merge but ambiguous rows keep the
    summary blocked.
  - Covers merge audit output for duplicate base manifests.
  - Covers merge audit output for ambiguous file-name fallback rows.
- Updated Phase 6 TODO.

## Claude Code Review

Claude Code is available locally as `/Users/chouhua/.local/bin/claude` and was
used as a read-only reviewer for targeted snippets. Tools were disabled and no
secrets or environment values were sent.

Claude Code found no clean-base merge regression. It confirmed the duplicate
base guard runs before matching/writeback and that audit rows now make the
blocked state explicit. It also identified file-name fallback ambiguity for
duplicate base file names; this slice now blocks that row-level ambiguity unless
the review row carries a precise matching `relative_path`.

## Release Impact

No labels are approved and no release thresholds are changed. This prevents
ambiguous benchmark manifest rows or ambiguous file-name fallback rows from
receiving reviewed manufacturing labels through the approved merge path.

## Remaining Work

- Populate real reviewed source, payload, and `details.*` labels.
- Keep base review and benchmark manifests deduplicated before apply/merge.
- Apply and merge only preflight-ready reviewer outputs.
- Tune thresholds after the reviewed release benchmark set is stable.
