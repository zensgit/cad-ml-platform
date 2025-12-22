# Day 1 - Stability and Reproducibility

Date: 2025-12-21

## Scope

- Enhance CAD preview smoke test to record render logs and improve reproducibility.
- Ensure report output is deterministic and avoids null-byte log issues.

## Changes

- Updated smoke test script:
  - `Athena/scripts/smoke_test_cad_preview.sh`
  - Added render log tail capture with null-byte stripping.
  - Added base64 newline check in preview payload.
  - Report includes render log snippet.

## Verification

- Command:
  - `/Users/huazhou/Downloads/Github/Athena/scripts/smoke_test_cad_preview.sh`
- Result:
  - `supported=true`, `pageCount=1`
  - Thumbnail PNG generated.
- Report generated:
  - `Athena/docs/SMOKE_CAD_PREVIEW_20251221_214440.md`

## Notes

- Render log tail is sanitized with `tr -d '\\000'` to avoid null-byte warnings.
