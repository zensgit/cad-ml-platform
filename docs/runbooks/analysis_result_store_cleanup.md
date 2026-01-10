# Analysis Result Store Cleanup

## Symptoms
- `analysis_result_store_files` keeps growing over time.
- Alerts for `AnalysisResultStoreGrowing`, `AnalysisResultCleanupDisabled`, or `AnalysisResultCleanupSkipped`.

## Checks
1. Confirm the store path:
   - `ANALYSIS_RESULT_STORE_DIR` is set and writable.
2. Verify cleanup policy:
   - `ANALYSIS_RESULT_STORE_TTL_SECONDS` and/or `ANALYSIS_RESULT_STORE_MAX_FILES`.
3. Verify cleanup loop (if enabled):
   - `ANALYSIS_RESULT_CLEANUP_INTERVAL_SECONDS` > 0.
4. Check maintenance stats:
   - `GET /api/v1/maintenance/stats` -> `analysis_result_store`.

## Remediation
1. Run a dry-run cleanup to see what would be removed:
   - `DELETE /api/v1/maintenance/analysis-results?dry_run=true&verbose=true`
2. Run cleanup without dry-run if the preview looks correct:
   - `DELETE /api/v1/maintenance/analysis-results`
3. If cleanup is disabled or skipped:
   - Set `ANALYSIS_RESULT_STORE_TTL_SECONDS` or `ANALYSIS_RESULT_STORE_MAX_FILES`.
   - Optionally enable the cleanup loop with `ANALYSIS_RESULT_CLEANUP_INTERVAL_SECONDS`.

## Notes
- Cleanup does not affect cache entries; cache will naturally expire.
- If the store grows rapidly, verify that analysis results are not overly large or duplicated.
