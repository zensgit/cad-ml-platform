"""Prediction confidence monitor for HybridClassifier (B5.3).

Tracks inference confidence distribution in a sliding window and fires
log-based alerts when drift patterns are detected.

Design principles:
  - Lock-free deque for thread safety under GIL (no asyncio required)
  - Alert de-duplication: same alert key suppressed within ALERT_COOLDOWN_SEC
  - No numpy dependency (stdlib only)
  - summary() dict is JSON-serialisable for any log/monitoring sidecar

Typical usage::

    monitor = PredictionMonitor()

    # Inside HybridClassifier.classify():
    monitor.record(
        predicted_class=result.label or "unknown",
        top1_confidence=result.confidence,
        confidence_margin=margin,
        text_hit=text_hit,
        filename_used=filename_pred is not None,
        latency_ms=latency_ms,
    )
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Immutable record of a single inference result."""

    timestamp: float
    predicted_class: str
    top1_confidence: float
    confidence_margin: float   # top1 − top2 probability gap
    text_hit: bool             # TextContentClassifier had ≥1 keyword match
    filename_used: bool        # FilenameClassifier contributed to fusion
    latency_ms: float


class PredictionMonitor:
    """Sliding-window inference monitor with drift detection.

    Args:
        window_size: Maximum records to keep in the sliding window (default 1000).
        low_conf_threshold: Confidence below which a prediction is "low confidence"
            (default 0.60).
        drift_alert_rate: low_conf_rate fraction that triggers a WARN alert
            (default 0.10 = 10 %).
        text_hit_alert_rate: text_hit_rate fraction below which an INFO alert fires
            to warn of silent text-signal loss (default 0.05 = 5 %).
        alert_cooldown_sec: Minimum seconds between repeated alerts of the same
            type (default 300 s = 5 min).
        min_window_for_alerts: Minimum records before drift checks run
            (default 100) — avoids false alarms on cold start.
    """

    LOW_CONF_THRESHOLD: float = 0.60
    DRIFT_ALERT_RATE: float = 0.10
    TEXT_HIT_ALERT_RATE: float = 0.05
    ALERT_COOLDOWN_SEC: int = 300
    MIN_WINDOW_FOR_ALERTS: int = 100

    def __init__(
        self,
        window_size: int = 1000,
        low_conf_threshold: float = LOW_CONF_THRESHOLD,
        drift_alert_rate: float = DRIFT_ALERT_RATE,
        text_hit_alert_rate: float = TEXT_HIT_ALERT_RATE,
        alert_cooldown_sec: int = ALERT_COOLDOWN_SEC,
        min_window_for_alerts: int = MIN_WINDOW_FOR_ALERTS,
    ) -> None:
        self.window_size = window_size
        self.low_conf_threshold = low_conf_threshold
        self.drift_alert_rate = drift_alert_rate
        self.text_hit_alert_rate = text_hit_alert_rate
        self.alert_cooldown_sec = alert_cooldown_sec
        self.min_window_for_alerts = min_window_for_alerts

        self._records: deque[PredictionRecord] = deque(maxlen=window_size)
        self._last_alert_time: Dict[str, Optional[float]] = {}

    # ── Record ────────────────────────────────────────────────────────────────

    def record(
        self,
        predicted_class: str,
        top1_confidence: float,
        confidence_margin: float = 0.0,
        text_hit: bool = False,
        filename_used: bool = True,
        latency_ms: float = 0.0,
    ) -> None:
        """Record one inference result and trigger drift checks.

        Args:
            predicted_class: The top-1 predicted class label.
            top1_confidence: Top-1 probability (0–1).
            confidence_margin: top1 − top2 probability gap (0–1).
            text_hit: True if TextContentClassifier returned any keyword match.
            filename_used: True if FilenameClassifier contributed to the result.
            latency_ms: End-to-end inference latency in milliseconds.
        """
        self._records.append(
            PredictionRecord(
                timestamp=time.monotonic(),
                predicted_class=predicted_class,
                top1_confidence=float(top1_confidence),
                confidence_margin=float(confidence_margin),
                text_hit=bool(text_hit),
                filename_used=bool(filename_used),
                latency_ms=float(latency_ms),
            )
        )
        self._check_drift()

    # ── Window statistics ─────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        """Number of records in the current window."""
        return len(self._records)

    @property
    def low_conf_rate(self) -> float:
        """Fraction of window records with top1_confidence < low_conf_threshold."""
        if not self._records:
            return 0.0
        return (
            sum(1 for r in self._records if r.top1_confidence < self.low_conf_threshold)
            / self.n
        )

    @property
    def text_hit_rate(self) -> float:
        """Fraction of window records where TextContentClassifier had a hit."""
        if not self._records:
            return 0.0
        return sum(1 for r in self._records if r.text_hit) / self.n

    @property
    def filename_used_rate(self) -> float:
        """Fraction of window records where FilenameClassifier was used."""
        if not self._records:
            return 0.0
        return sum(1 for r in self._records if r.filename_used) / self.n

    @property
    def avg_confidence(self) -> float:
        """Mean top1_confidence over the window."""
        if not self._records:
            return 0.0
        return statistics.mean(r.top1_confidence for r in self._records)

    @property
    def avg_margin(self) -> float:
        """Mean confidence_margin over the window."""
        if not self._records:
            return 0.0
        return statistics.mean(r.confidence_margin for r in self._records)

    @property
    def avg_latency_ms(self) -> float:
        """Mean inference latency over the window."""
        if not self._records:
            return 0.0
        return statistics.mean(r.latency_ms for r in self._records)

    @property
    def p95_latency_ms(self) -> float:
        """95th-percentile inference latency over the window."""
        if not self._records:
            return 0.0
        sorted_lat = sorted(r.latency_ms for r in self._records)
        idx = max(0, int(0.95 * len(sorted_lat)) - 1)
        return sorted_lat[idx]

    @property
    def class_distribution(self) -> List[Tuple[str, float]]:
        """Top-10 (class, fraction) pairs sorted by frequency descending."""
        if not self._records:
            return []
        counts = Counter(r.predicted_class for r in self._records)
        return [(cls, count / self.n) for cls, count in counts.most_common(10)]

    # ── Drift detection ───────────────────────────────────────────────────────

    def _can_alert(self, key: str) -> bool:
        """Return True and update timestamp if this alert is not in cooldown.

        Uses None as the sentinel for "never fired" so the first call always
        returns True regardless of system uptime (avoids false suppression when
        time.monotonic() < alert_cooldown_sec after a fresh boot).
        """
        now = time.monotonic()
        last = self._last_alert_time.get(key)
        if last is None or now - last >= self.alert_cooldown_sec:
            self._last_alert_time[key] = now
            return True
        return False

    def _check_drift(self) -> None:
        """Run all drift checks; fires log alerts when thresholds are exceeded."""
        if self.n < self.min_window_for_alerts:
            return

        # ① Low-confidence drift
        lc_rate = self.low_conf_rate
        if lc_rate > self.drift_alert_rate and self._can_alert("low_conf"):
            logger.warning(
                "DRIFT ALERT [low_conf]: %.1f%% of last %d predictions have "
                "confidence < %.0f%%  (threshold: %.0f%%)  "
                "avg_conf=%.3f  avg_margin=%.3f",
                lc_rate * 100,
                self.n,
                self.low_conf_threshold * 100,
                self.drift_alert_rate * 100,
                self.avg_confidence,
                self.avg_margin,
            )

        # ② Text-signal loss
        txt_rate = self.text_hit_rate
        if txt_rate < self.text_hit_alert_rate and self._can_alert("text_hit_loss"):
            logger.info(
                "TEXT SIGNAL: only %.1f%% of last %d predictions have keyword hits "
                "(expected ~15%% on typical DXF batches)  "
                "— possible text extraction regression or dataset shift",
                txt_rate * 100,
                self.n,
            )

    def check_drift(self) -> bool:
        """Explicitly check for drift and return True if any threshold is exceeded.

        Does not fire log alerts (use for health-check endpoints).
        """
        if self.n < self.min_window_for_alerts:
            return False
        if self.low_conf_rate > self.drift_alert_rate:
            return True
        if self.text_hit_rate < self.text_hit_alert_rate:
            return True
        return False

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        """Return a JSON-serialisable monitoring snapshot."""
        return {
            "window_size": self.window_size,
            "n": self.n,
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_margin": round(self.avg_margin, 4),
            "low_conf_rate": round(self.low_conf_rate, 4),
            "low_conf_threshold": self.low_conf_threshold,
            "text_hit_rate": round(self.text_hit_rate, 4),
            "filename_used_rate": round(self.filename_used_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "top5_classes": [
                {"class": cls, "fraction": round(frac, 4)}
                for cls, frac in self.class_distribution[:5]
            ],
            "drift_detected": self.check_drift(),
        }

    def reset(self) -> None:
        """Clear all records and alert timestamps (useful for test isolation)."""
        self._records.clear()
        self._last_alert_time.clear()
