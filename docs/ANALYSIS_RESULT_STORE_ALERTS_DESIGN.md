# Analysis Result Store Alerts Design

## Overview
This design adds alerts for analysis result disk usage and cleanup configuration
so operators can detect missing retention policies and growth in stored files.

## Metrics Used
- `analysis_result_store_files`: gauge of on-disk analysis result count.
- `analysis_result_cleanup_total{status=...}`: cleanup attempts by status.

## Alert Rules
1. AnalysisResultStoreGrowing
   - Trigger: `analysis_result_store_files > 10000` for 30m.
   - Purpose: detect unbounded growth of stored results.

2. AnalysisResultCleanupDisabled
   - Trigger: `analysis_result_store_files > 0` and
     `increase(analysis_result_cleanup_total{status="disabled"}[1h]) > 0` for 30m.
   - Purpose: detect cleanup attempts while the store is disabled.

3. AnalysisResultCleanupSkipped
   - Trigger: `analysis_result_store_files > 0` and
     `increase(analysis_result_cleanup_total{status="skipped"}[1h]) > 0` for 30m.
   - Purpose: detect cleanup attempts without a retention policy.

## Runbook
- `docs/runbooks/analysis_result_store_cleanup.md`

## Threshold Guidance
- Adjust `10000` based on storage capacity and average result size.
- If cleanup is intentionally disabled, silence the two cleanup alerts.
