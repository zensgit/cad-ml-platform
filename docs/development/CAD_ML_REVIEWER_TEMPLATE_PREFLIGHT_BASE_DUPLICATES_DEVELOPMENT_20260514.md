# CAD ML Reviewer Template Preflight Base Duplicates Development

Date: 2026-05-14

## Goal

Make reviewer-template preflight reject ambiguous base review manifests before
apply. If the supplied base manifest contains duplicate row identities, apply can
otherwise match a returned reviewer row to an arbitrary first row. This slice
surfaces that artifact problem explicitly.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - Adds duplicate identity detection for supplied base manifests.
  - Preflight summaries now include:
    - `base_manifest_duplicate_identity_count`
    - `base_manifest_duplicate_identifiers`
  - Adds `base_manifest_duplicate_rows` to blocking reasons when duplicate base
    row identities are present.
  - Preflight Markdown now shows duplicate base identity count and lists the
    duplicate base row IDs.
- Updated tests.
  - Covers duplicate base manifest summary blocking.
  - Covers duplicate base manifest Markdown output.
  - Keeps no-base default behavior explicitly at duplicate count `0`.
- Updated Phase 6 TODO.

## Claude Code Review

Claude Code is available locally as `/Users/chouhua/.local/bin/claude` and was
used as a read-only reviewer for the targeted snippets. Tools were disabled and
only local code/test excerpts were sent.

Claude Code found no default-path regression. It noted that duplicate-base
blocking is additive and only activates when a base manifest is supplied and
contains duplicate row identities.

## Release Impact

No labels are approved and no release threshold changes. This makes reviewer
template application safer by requiring the active review manifest to be a
non-ambiguous base before preflight can pass.

## Remaining Work

- Populate real reviewed source, payload, and `details.*` labels.
- Preflight returned templates against the active, non-ambiguous review manifest.
- Apply only preflight-ready templates.
- Tune thresholds after the reviewed release benchmark set is stable.
