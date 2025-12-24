"""Background task to detect orphan vectors (no cached analysis result)."""

from __future__ import annotations

import asyncio
import os
import time

from src.utils.analysis_metrics import vector_orphan_scan_last_seconds, vector_orphan_total
from src.utils.cache import get_client


async def orphan_scan_loop(interval: float | None = None) -> None:  # pragma: no cover (loop)
    scan_interval = interval or float(os.getenv("VECTOR_ORPHAN_SCAN_INTERVAL_SECONDS", "300"))
    last_scan = time.time()
    while True:
        try:
            from src.core.similarity import _VECTOR_STORE  # type: ignore

            client = get_client()
            orphan_count = 0
            if client is not None:
                for vid in list(_VECTOR_STORE.keys()):
                    try:
                        key = f"analysis_result:{vid}"
                        raw = await client.get(key)  # type: ignore[attr-defined]
                        if raw is None:
                            orphan_count += 1
                    except Exception:
                        continue
            # Increment metric once per scan (not per vector) to avoid explosion; use count value as inc
            if orphan_count > 0:
                vector_orphan_total.inc(orphan_count)
            vector_orphan_scan_last_seconds.set(0)
            last_scan = time.time()
        except Exception:
            pass
        # update age gauge periodically
        try:
            vector_orphan_scan_last_seconds.set(time.time() - last_scan)
        except Exception:
            pass
        await asyncio.sleep(scan_interval)


__all__ = ["orphan_scan_loop"]
