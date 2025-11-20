"""Safe wrappers around Prometheus metric operations.

These helpers prevent metric failures from impacting business logic.
"""

from __future__ import annotations

from typing import Any


def safe_inc(counter: Any, **labels: str) -> None:
    try:
        counter.labels(**labels).inc() if labels else counter.inc()
    except Exception:
        pass


def safe_observe(hist: Any, value: float, **labels: str) -> None:
    try:
        hist.labels(**labels).observe(value) if labels else hist.observe(value)
    except Exception:
        pass


def safe_set(gauge: Any, value: float, **labels: str) -> None:
    try:
        gauge.labels(**labels).set(value) if labels else gauge.set(value)
    except Exception:
        pass
