# Faiss Degradation Flapping Runbook

Purpose: triage and mitigate frequent degraded/restored toggles.

Key metrics:
- `similarity_degraded_total{event}` — degraded/restored counters
- `faiss_degraded_duration_seconds` — current degraded duration
- `faiss_next_recovery_eta_seconds` — next auto-recovery ETA gauge
- `faiss_recovery_attempts_total{result}` — success/error attempts
- `faiss_recovery_suppressed_total{reason}` — suppressed due to flapping
- `faiss_recovery_suppression_remaining_seconds` — remaining suppression window seconds (0 if none)

Health API:
- `GET /api/v1/health/faiss/health` — returns `degraded`, `reason`, `degradation_history_count`, `next_recovery_eta`, `manual_recovery_in_progress`.
- `POST /api/v1/faiss/recover` — manual recovery (sets `manual_recovery_in_progress`).

Env knobs:
- `FAISS_RECOVERY_FLAP_THRESHOLD` — max toggles in window to trigger suppression.
- `FAISS_RECOVERY_FLAP_WINDOW_SECONDS` — window length for flap detection.
- `FAISS_RECOVERY_SUPPRESSION_SECONDS` — suppression period once flapping detected.
- `FAISS_RECOVERY_STATE_PATH` — persistence file path for recovery state.

Procedure:
1. Confirm flapping via Grafana panel and health endpoint history count.
2. Enable suppression by tuning env knobs (`FLAP_THRESHOLD` smaller, `SUPPRESSION_SECONDS` longer).
3. Trigger `POST /api/v1/faiss/recover` if degraded and suppression lifted.
4. Verify `faiss_next_recovery_eta_seconds` updates and resets to 0 on recovery.
5. Investigate root cause (backend availability, index corruption, IO). Consider switching to memory backend temporarily.
6. If suppression window appears stuck (remaining seconds not decreasing) check:
   - Recovery loop thread alive
   - System clock monotonicity
   - Recent exceptions in logs around recovery
   - `FAISS_RECOVERY_SUPPRESSION_SECONDS` overly large
   - Fallback: manually clear suppression via short process restart or env tweak

Rollback guidance:
- If changes cause instability, revert env knob tweaks and disable manual overrides.
- Use cache rollback endpoints for unrelated cache tuning changes.

References:
- `src/core/similarity.py` — recovery loop, suppression, persistence.
- `src/api/v1/health.py` — health/recover endpoints.
- `src/utils/analysis_metrics.py` — metrics definitions.
 - Prometheus rule `RecoverySuppressionWindowStuck` — detects non-decreasing suppression remaining seconds.
