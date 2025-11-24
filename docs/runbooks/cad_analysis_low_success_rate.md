## CAD Analysis Low Success Rate Runbook

### Trigger
Alert `CadAnalysisLowSuccessRate` fires when success rate <90% for 5m.

### Immediate Actions
1. Check Grafana panel for `analysis_requests_total` breakdown by status.
2. Inspect recent `analysis_error_code_total` spikes (codes INTERNAL_ERROR, INPUT_VALIDATION).
3. Review deployment history for last 30 minutes.

### Diagnostic Steps
1. Sample logs for failing requests (trace IDs if available).
2. Verify adapter parse times; ensure no large inputs exceeding limits.
3. Confirm Redis availability (if backend = redis) for vector operations.

### Remediation
1. Roll back recent deployment if correlated.
2. Adjust `ANALYSIS_MAX_FILE_MB` or `ANALYSIS_MAX_ENTITIES` if legitimate workload exceeds limits.
3. Patch failing rule logic or add fallback try/except around new code paths.

### Preventative
Add tests for new error code scenarios before deploying.

