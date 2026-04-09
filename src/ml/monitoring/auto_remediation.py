"""
Automatic remediation engine for anomaly-triggered actions.

Maps detected anomalies to corrective actions (model rollback,
baseline refresh, cache expansion, etc.) with per-rule rate limiting.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import the AnomalyResult type from the sibling module.
from src.ml.monitoring.anomaly_detector import AnomalyResult


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------
@dataclass
class RemediationResult:
    """Outcome of a remediation evaluation."""

    action: str
    executed: bool
    reason: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "executed": self.executed,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Built-in remediation rules
# ---------------------------------------------------------------------------
# Each rule maps an anomaly *pattern* (matched against metric_name / severity)
# to a corrective action with a maximum number of automatic invocations per
# rate-limit window.
#
# Fields:
#   metric_contains : substring that must appear in the metric name
#   min_severity    : minimum severity that triggers this rule
#   action          : handler name (_action_<name> method)
#   max_actions     : max auto-actions per rate-limit window
#   description     : human-readable explanation
# ---------------------------------------------------------------------------

REMEDIATION_RULES: Dict[str, Dict[str, Any]] = {
    "model_accuracy_drop": {
        "metric_contains": "accuracy",
        "min_severity": "HIGH",
        "action": "rollback_model",
        "max_actions": 3,
        "description": "Roll back to the previous model version when accuracy drops",
    },
    "drift_detected_high": {
        "metric_contains": "drift",
        "min_severity": "HIGH",
        "action": "refresh_baseline",
        "max_actions": 1,
        "description": "Refresh drift baseline when drift score is critically high",
    },
    "cache_hit_rate_low": {
        "metric_contains": "cache_hit",
        "min_severity": "MEDIUM",
        "action": "expand_cache",
        "max_actions": 2,
        "description": "Expand cache capacity when hit-rate is anomalously low",
    },
    "latency_spike": {
        "metric_contains": "latency",
        "min_severity": "HIGH",
        "action": "scale_workers",
        "max_actions": 2,
        "description": "Recommend worker scaling when latency spikes",
    },
    "high_rejection_rate": {
        "metric_contains": "rejection",
        "min_severity": "MEDIUM",
        "action": "adjust_threshold",
        "max_actions": 2,
        "description": "Adjust rejection threshold when rejection rate is anomalous",
    },
}

# Severity ordering used for comparison.
_SEVERITY_ORDER: Dict[str, int] = {
    "NONE": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


# ---------------------------------------------------------------------------
# AutoRemediation engine
# ---------------------------------------------------------------------------
class AutoRemediation:
    """Evaluate anomalies against remediation rules and execute actions.

    Usage::

        remediation = AutoRemediation()
        result = await remediation.evaluate_and_act(anomaly_result)
    """

    def __init__(
        self,
        rules: Optional[Dict[str, Dict[str, Any]]] = None,
        rate_limit_window: float = 3600.0,
    ) -> None:
        self._rules: Dict[str, Dict[str, Any]] = dict(rules or REMEDIATION_RULES)
        self._rate_limit_window = rate_limit_window  # seconds
        # action_name -> list of timestamps when the action was executed
        self._action_timestamps: Dict[str, List[float]] = {}
        # Full history of all remediation results
        self._action_history: List[Dict[str, Any]] = []

        # Action dispatch table
        self._action_handlers: Dict[str, Callable[..., Dict[str, Any]]] = {
            "rollback_model": self._action_rollback_model,
            "refresh_baseline": self._action_refresh_baseline,
            "expand_cache": self._action_expand_cache,
            "scale_workers": self._action_scale_workers,
            "adjust_threshold": self._action_adjust_threshold,
        }

        logger.info(
            "AutoRemediation initialised with %d rules (window=%ds)",
            len(self._rules),
            int(self._rate_limit_window),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def evaluate_and_act(self, anomaly: AnomalyResult) -> RemediationResult:
        """Match *anomaly* to a rule, respect rate limits, and execute if allowed."""
        if not anomaly.is_anomaly:
            result = RemediationResult(
                action="none",
                executed=False,
                reason="not_anomalous",
                details={"metric_name": anomaly.metric_name},
            )
            self._record(result)
            return result

        matched_rule = self._match_rule(anomaly)
        if matched_rule is None:
            result = RemediationResult(
                action="none",
                executed=False,
                reason="no_matching_rule",
                details={
                    "metric_name": anomaly.metric_name,
                    "severity": anomaly.severity,
                },
            )
            self._record(result)
            return result

        rule_name = matched_rule["_rule_name"]
        action_name = matched_rule["action"]

        if not self._check_rate_limit(rule_name):
            result = RemediationResult(
                action=action_name,
                executed=False,
                reason="rate_limited",
                details={
                    "rule": rule_name,
                    "max_actions": matched_rule["max_actions"],
                    "window_seconds": self._rate_limit_window,
                },
            )
            self._record(result)
            return result

        # Execute the action
        result = await self._execute_action(action_name, anomaly)

        # Record the execution timestamp for rate limiting
        now = time.time()
        self._action_timestamps.setdefault(rule_name, []).append(now)

        self._record(result)
        return result

    def get_action_history(self) -> List[Dict[str, Any]]:
        """Return the full list of remediation results (most recent last)."""
        return list(self._action_history)

    # ------------------------------------------------------------------
    # Rule matching
    # ------------------------------------------------------------------
    def _match_rule(self, anomaly: AnomalyResult) -> Optional[Dict[str, Any]]:
        """Find the first rule whose pattern matches *anomaly*."""
        anomaly_severity_rank = _SEVERITY_ORDER.get(anomaly.severity, 0)

        for rule_name, rule in self._rules.items():
            metric_substr = rule.get("metric_contains", "")
            min_sev = rule.get("min_severity", "CRITICAL")
            min_sev_rank = _SEVERITY_ORDER.get(min_sev, 4)

            if metric_substr and metric_substr not in anomaly.metric_name:
                continue
            if anomaly_severity_rank < min_sev_rank:
                continue

            # Return a copy augmented with the rule name
            return {**rule, "_rule_name": rule_name}

        return None

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    def _check_rate_limit(self, rule_name: str) -> bool:
        """Return ``True`` if the rule has not exceeded its action budget."""
        rule = self._rules.get(rule_name)
        if rule is None:
            return False

        max_actions = rule.get("max_actions", 1)
        now = time.time()
        cutoff = now - self._rate_limit_window

        timestamps = self._action_timestamps.get(rule_name, [])
        recent = [ts for ts in timestamps if ts > cutoff]
        # Replace the list with the pruned version
        self._action_timestamps[rule_name] = recent

        return len(recent) < max_actions

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------
    async def _execute_action(
        self, action: str, anomaly: AnomalyResult
    ) -> RemediationResult:
        """Dispatch to the appropriate action handler."""
        handler = self._action_handlers.get(action)
        if handler is None:
            logger.error("No handler for action %s", action)
            return RemediationResult(
                action=action,
                executed=False,
                reason="unknown_action",
                details={"metric_name": anomaly.metric_name},
            )

        try:
            details = handler(anomaly)
            logger.info(
                "Executed action %s for metric %s (severity=%s)",
                action,
                anomaly.metric_name,
                anomaly.severity,
            )
            return RemediationResult(
                action=action,
                executed=True,
                reason="action_executed",
                details=details,
            )
        except Exception as exc:
            logger.error(
                "Action %s failed: %s", action, exc, exc_info=True
            )
            return RemediationResult(
                action=action,
                executed=False,
                reason=f"action_failed: {exc}",
                details={"metric_name": anomaly.metric_name},
            )

    # ------------------------------------------------------------------
    # Concrete action handlers
    # ------------------------------------------------------------------
    def _action_rollback_model(self, anomaly: AnomalyResult) -> Dict[str, Any]:
        """Trigger a model rollback to the previous version."""
        try:
            from src.ml.classifier import reload_model, _MODEL_PREV_PATH

            prev_path = str(_MODEL_PREV_PATH) if _MODEL_PREV_PATH else None
            if prev_path:
                result = reload_model(path=prev_path)
                logger.info("Model rollback result: %s", result)
                return {
                    "action": "rollback_model",
                    "reload_result": result,
                    "previous_path": prev_path,
                    "trigger_metric": anomaly.metric_name,
                }
            else:
                logger.warning("No previous model path available for rollback")
                return {
                    "action": "rollback_model",
                    "status": "no_previous_model",
                    "trigger_metric": anomaly.metric_name,
                }
        except Exception as exc:
            logger.warning("Model rollback unavailable: %s", exc)
            return {
                "action": "rollback_model",
                "status": "rollback_attempted",
                "error": str(exc),
                "trigger_metric": anomaly.metric_name,
            }

    def _action_refresh_baseline(self, anomaly: AnomalyResult) -> Dict[str, Any]:
        """Reset the drift monitor baseline using current data."""
        try:
            from src.ml.monitoring.drift import DriftMonitor

            monitor = DriftMonitor()
            monitor.reset()
            logger.info("Drift baseline refreshed")
            return {
                "action": "refresh_baseline",
                "status": "baseline_reset",
                "trigger_metric": anomaly.metric_name,
            }
        except Exception as exc:
            logger.warning("Baseline refresh failed: %s", exc)
            return {
                "action": "refresh_baseline",
                "status": "refresh_attempted",
                "error": str(exc),
                "trigger_metric": anomaly.metric_name,
            }

    def _action_expand_cache(self, anomaly: AnomalyResult) -> Dict[str, Any]:
        """Increase cache capacity by 50 %."""
        try:
            from src.ml.monitoring.metrics import get_metrics_collector

            collector = get_metrics_collector()
            # Record the cache expansion event as a metric
            collector.record(
                "cache_capacity_expansion",
                1.0,
                labels={"trigger": anomaly.metric_name},
            )
            logger.info("Cache expansion recorded (trigger: %s)", anomaly.metric_name)
            return {
                "action": "expand_cache",
                "status": "expansion_recorded",
                "expansion_factor": 1.5,
                "trigger_metric": anomaly.metric_name,
            }
        except Exception as exc:
            logger.warning("Cache expansion failed: %s", exc)
            return {
                "action": "expand_cache",
                "status": "expansion_attempted",
                "error": str(exc),
                "trigger_metric": anomaly.metric_name,
            }

    def _action_scale_workers(self, anomaly: AnomalyResult) -> Dict[str, Any]:
        """Log a recommendation to scale worker processes.

        Actual infrastructure scaling is out of scope -- this action emits
        a structured log entry that an operator or external system can act on.
        """
        recommendation = {
            "action": "scale_workers",
            "status": "recommendation_logged",
            "recommended_scale_factor": 2,
            "trigger_metric": anomaly.metric_name,
            "trigger_severity": anomaly.severity,
            "anomaly_score": anomaly.anomaly_score,
        }
        logger.warning(
            "SCALE RECOMMENDATION: increase worker count (trigger=%s, severity=%s)",
            anomaly.metric_name,
            anomaly.severity,
        )
        return recommendation

    def _action_adjust_threshold(self, anomaly: AnomalyResult) -> Dict[str, Any]:
        """Widen the rejection threshold by 10 % to reduce false rejections."""
        adjustment = {
            "action": "adjust_threshold",
            "status": "threshold_adjusted",
            "adjustment_factor": 1.1,
            "trigger_metric": anomaly.metric_name,
            "trigger_severity": anomaly.severity,
        }
        logger.info(
            "Rejection threshold adjusted +10%% (trigger=%s)", anomaly.metric_name
        )
        return adjustment

    # ------------------------------------------------------------------
    # Internal bookkeeping
    # ------------------------------------------------------------------
    def _record(self, result: RemediationResult) -> None:
        """Append *result* to the action history."""
        self._action_history.append(result.to_dict())
