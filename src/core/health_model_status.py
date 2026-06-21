"""Pure status/uptime/snapshot computation for the model_health endpoint.

Extracted from src/api/v1/health.py (behavior-preserving router slimming). The
handler keeps the FastAPI wiring, Prometheus side effects, and response model;
it calls this pure function and is re-exported for testing.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


def compute_model_health(
    info: Dict[str, Any], *, now: Optional[float] = None
) -> Dict[str, Any]:
    """Derive status / rollback level / snapshot count / uptime from model info.

    Verbatim port of the prior inline handler logic. `now` is injectable for
    deterministic tests; it defaults to time.time() to preserve behaviour.
    """
    rollback_level = info.get("rollback_level", 0)
    if not info.get("loaded"):
        status = "absent"
    elif rollback_level > 0:
        status = "rollback"
    else:
        status = "ok"

    snapshots_available = sum(
        1
        for flag in (
            info.get("has_prev"),
            info.get("has_prev2"),
            info.get("has_prev3"),
        )
        if flag
    )

    uptime: Optional[float] = None
    loaded_at = info.get("loaded_at")
    if loaded_at:
        try:
            base = now if now is not None else time.time()
            uptime = base - float(loaded_at)
        except Exception:
            uptime = None

    return {
        "status": status,
        "rollback_level": rollback_level,
        "snapshots_available": snapshots_available,
        "uptime_seconds": uptime,
    }
