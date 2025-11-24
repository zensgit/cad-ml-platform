## CAD Analysis Entity Rejection Spike Runbook

### Trigger
Alert `CadAnalysisEntityRejectionSpike` fires when entity count rejections increase by >50 in 10m.

### Immediate Actions
1. Inspect traffic source (batch import vs normal usage).
2. Confirm `ANALYSIS_MAX_ENTITIES` value appropriateness.
3. Validate rejects are legitimate (not parser miscount).

### Diagnostic Steps
1. Sample rejected files; compute actual entity counts via offline tools.
2. Check if new partner integration sending oversized assemblies.
3. Review logs for adapter warnings or partial parses.

### Remediation
1. Raise limit temporarily if legitimate business need.
2. Add pre-ingestion advisory to clients (split assemblies).
3. Optimize entity counting code path to avoid over-count (dedupe).

### Preventative
Introduce adaptive limit: dynamic threshold based on CPU load and queue depth.

