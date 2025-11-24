## CAD Analysis Internal Errors Runbook

### Trigger
Alert `CadAnalysisHighInternalErrors` fires when INTERNAL_ERROR code rate >0.2/s for 5m.

### Immediate Actions
1. Triage stack traces for recurring exception types.
2. Check recent code changes touching `analyze.py`, `similarity.py`, adapters.
3. Confirm dependency versions; ensure no missing optional libs causing exceptions.

### Diagnostic Steps
1. Identify dominant stage: correlate with `analysis_stage_duration_seconds` and logs.
2. Inspect Redis connectivity if vector operations involved.
3. Validate YAML process rules loading; check hash via `/process/rules/audit`.

### Remediation
1. Hotfix null checks or broaden exception handling around failing stage.
2. If adapter parsing: add format-specific guard, reduce file size limit temporarily.
3. Disable problematic feature via env (e.g., skip similarity) until fix deployed.

### Preventative
Expand unit tests for edge cases uncovered; add synthetic failing input to regression suite.

