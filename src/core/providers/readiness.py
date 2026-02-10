"""Provider readiness helpers.

These helpers are used by runtime readiness probes (/ready) and health payloads
to reason about optional provider availability without hard-coding business
logic into API handlers.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.core.providers.bootstrap import bootstrap_core_provider_registry
from src.core.providers.registry import ProviderRegistry
from src.utils.metrics import (
    core_provider_check_duration_seconds,
    core_provider_checks_total,
)

ProviderId = Tuple[str, str]  # (domain, provider_name)


def parse_provider_id_list(raw: str) -> List[ProviderId]:
    """Parse comma/space separated provider IDs into (domain, provider) tuples.

    Accepts tokens like:
    - "classifier/hybrid"
    - "ocr:paddle"

    Invalid tokens are ignored.
    """
    if not raw:
        return []
    tokens = [t.strip() for t in re.split(r"[,\s]+", raw) if t.strip()]
    results: List[ProviderId] = []
    for token in tokens:
        domain = ""
        name = ""
        if "/" in token:
            domain, name = token.split("/", 1)
        elif ":" in token:
            domain, name = token.split(":", 1)
        domain = domain.strip()
        name = name.strip()
        if not domain or not name:
            continue
        results.append((domain, name))
    return results


def format_provider_id(provider_id: ProviderId) -> str:
    domain, name = provider_id
    return f"{domain}/{name}"


@dataclass
class ProviderReadinessItem:
    id: str
    ready: bool
    error: Optional[str] = None
    checked_at: Optional[float] = None
    latency_ms: Optional[float] = None


@dataclass
class ProviderReadinessSummary:
    ok: bool
    degraded: bool
    required: List[str]
    optional: List[str]
    required_down: List[str]
    optional_down: List[str]
    timeout_seconds: float
    checked_at: float
    results: List[ProviderReadinessItem]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "degraded": self.degraded,
            "required": list(self.required),
            "optional": list(self.optional),
            "required_down": list(self.required_down),
            "optional_down": list(self.optional_down),
            "timeout_seconds": float(self.timeout_seconds),
            "checked_at": float(self.checked_at),
            "results": [
                {
                    "id": item.id,
                    "ready": bool(item.ready),
                    "error": item.error,
                    "checked_at": item.checked_at,
                    "latency_ms": item.latency_ms,
                }
                for item in self.results
            ],
        }


async def check_provider_readiness(
    required: Sequence[ProviderId],
    optional: Sequence[ProviderId],
    timeout_seconds: float = 0.5,
) -> ProviderReadinessSummary:
    """Best-effort, timeout-bounded readiness check for selected providers."""
    bootstrap_core_provider_registry()

    if timeout_seconds <= 0:
        timeout_seconds = 0.5
    timeout_seconds = float(min(timeout_seconds, 10.0))

    now = time.time()
    required_ids = [format_provider_id(pid) for pid in required]
    optional_ids = [format_provider_id(pid) for pid in optional]

    async def _check_one(domain: str, name: str) -> ProviderReadinessItem:
        started_at = time.perf_counter()
        checked_at = time.time()
        provider_id = f"{domain}/{name}"
        try:
            provider = ProviderRegistry.get(domain, name)
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            try:
                core_provider_checks_total.labels(
                    source="readiness",
                    domain=domain,
                    provider=name,
                    result="init_error",
                ).inc()
                core_provider_check_duration_seconds.labels(
                    source="readiness",
                    domain=domain,
                    provider=name,
                ).observe(latency_ms / 1000.0)
            except Exception:
                pass
            return ProviderReadinessItem(
                id=provider_id,
                ready=False,
                error=f"init_error: {exc}",
                checked_at=checked_at,
                latency_ms=latency_ms,
            )

        # Standardize on the provider framework's health_check() so readiness
        # updates provider runtime status consistently with `/providers/health`.
        ok = await provider.health_check(timeout_seconds=timeout_seconds)  # type: ignore[arg-type]
        err = getattr(provider, "last_error", None)

        latency_ms = (time.perf_counter() - started_at) * 1000.0
        try:
            core_provider_checks_total.labels(
                source="readiness",
                domain=domain,
                provider=name,
                result="ready" if ok else "down",
            ).inc()
            core_provider_check_duration_seconds.labels(
                source="readiness",
                domain=domain,
                provider=name,
            ).observe(latency_ms / 1000.0)
        except Exception:
            pass
        return ProviderReadinessItem(
            id=provider_id,
            ready=bool(ok),
            error=err,
            checked_at=checked_at,
            latency_ms=latency_ms,
        )

    # De-dupe while preserving order
    seen: set[str] = set()
    provider_list: List[ProviderId] = []
    for pid in list(required) + list(optional):
        pid_str = format_provider_id(pid)
        if pid_str in seen:
            continue
        seen.add(pid_str)
        provider_list.append(pid)

    tasks = [_check_one(domain, name) for domain, name in provider_list]
    results = list(await asyncio.gather(*tasks)) if tasks else []

    ready_map = {item.id: item.ready for item in results}
    required_down = [pid for pid in required_ids if not ready_map.get(pid, False)]
    optional_down = [pid for pid in optional_ids if not ready_map.get(pid, False)]

    ok = not required_down
    degraded = ok and bool(optional_down)

    return ProviderReadinessSummary(
        ok=ok,
        degraded=degraded,
        required=required_ids,
        optional=optional_ids,
        required_down=required_down,
        optional_down=optional_down,
        timeout_seconds=timeout_seconds,
        checked_at=now,
        results=results,
    )


def load_provider_readiness_config_from_env() -> (
    Tuple[List[ProviderId], List[ProviderId]]
):
    """Load required/optional provider lists from environment variables."""
    required_raw = os.getenv("READINESS_REQUIRED_PROVIDERS", "").strip()
    optional_raw = os.getenv("READINESS_OPTIONAL_PROVIDERS", "").strip()
    return parse_provider_id_list(required_raw), parse_provider_id_list(optional_raw)
