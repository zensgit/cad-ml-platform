"""Pure status/age derivation for the faiss_health endpoint.

Extracted from src/api/v1/health.py (behavior-preserving router slimming). The
handler keeps the FastAPI wiring, module-global reads, Prometheus side effects,
and response model; it calls this pure function, which is re-exported from
health.py for compatibility and testing.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


def compute_faiss_health(
    *,
    available: bool,
    degraded: bool,
    last_export_ts: Optional[float],
    last_import: Optional[float],
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Derive FAISS health status + index age from raw signals.

    Verbatim port of the prior inline handler logic:

    * status priority is ``degraded > unavailable > ok``.
    * age is measured from the last export timestamp, falling back to the last
      import timestamp, and is ``None`` when neither is known.

    ``now`` is injectable for deterministic tests; it defaults to time.time()
    to preserve behaviour.
    """
    if degraded:
        status = "degraded"
    elif not available:
        status = "unavailable"
    else:
        status = "ok"

    base = now if now is not None else time.time()
    age_seconds: Optional[int] = None
    if last_export_ts:
        age_seconds = int(base - last_export_ts)
    elif last_import:
        age_seconds = int(base - last_import)

    return {"status": status, "age_seconds": age_seconds}
