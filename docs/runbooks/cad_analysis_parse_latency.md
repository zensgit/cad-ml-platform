## CAD Analysis Parse Latency Runbook

### Trigger
Alert `CadAnalysisStageLatencyParseP95High` fires when parse stage p95 >500ms for 10m.

### Immediate Actions
1. Confirm spike is global (multiple instances) vs single pod.
2. Check input sizes; look for unusually large DXF/STEP files.
3. Review CPU usage; ensure no resource starvation.

### Diagnostic Steps
1. Examine recent adapter code changes for performance regressions.
2. Compare `analysis_parse_latency_budget_ratio` distribution to target.
3. If STEP: confirm optional pythonocc not causing import delays; fallback logic working.
4. Profile a sample slow file locally (enable timing logs in adapter).

### Remediation
1. Implement early size/entity count rejection if missing.
2. Add incremental parsing (stream) for large formats where feasible.
3. Cache parse results for repeated identical files via hash key.

### Preventative
Add performance regression test measuring parse latency budgets for common formats.

