# Eval Reporting Phase 3 Release Baseline Hardening — Design

日期：2026-04-03

## Scope

Batch 19B: establish new release baseline after Phase 3 merge, with regression guards preventing snippet/prefill from reappearing and protecting kept surfaces.

## New Release Chain (post-merge)

```
dashboard_payload → release_draft_payload → publish_payload → publish_result
```

`release_note_snippet` and `release_draft_prefill` have been absorbed into `release_draft_payload`.

## Hardening Tests Added

6 new regression guard tests:

1. `test_merged_release_note_snippet_not_in_workflow` — negative guard (step names + sparse-checkout)
2. `test_merged_release_draft_prefill_not_in_workflow` — negative guard (step names + sparse-checkout)
3. `test_kept_release_draft_payload_still_present_after_merge` — positive guard
4. `test_kept_publish_payload_still_present_after_merge` — positive guard
5. `test_kept_publish_result_still_present_after_merge` — positive guard
6. `test_release_draft_payload_reads_dashboard_not_prefill` — input guard (--dashboard-payload-json, no --prefill-json/--snippet-json)

## What Was NOT Changed

- No new artifacts added
- No publish_payload → publish_result merge
- No publish_result contract changed
