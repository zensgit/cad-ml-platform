"""Knowledge bridge for the core provider framework.

This module adapts deterministic, built-in knowledge libraries into the generic
``src.core.providers`` abstractions so they can participate in:
- provider registry snapshots (`/api/v1/*/providers/registry`)
- best-effort health checks (`/api/v1/*/providers/health`)
- readiness selection (`/ready` with READINESS_*_PROVIDERS)

These providers are intentionally lightweight: they do not expose a full query
surface (that is handled by dedicated API routers under `src/api/v1/*`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.core.providers.base import BaseProvider, ProviderConfig
from src.core.providers.registry import ProviderRegistry


@dataclass
class KnowledgeProviderConfig(ProviderConfig):
    """Configuration for knowledge adapter providers."""

    provider_name: str = "unknown"


class ToleranceKnowledgeProviderAdapter(
    BaseProvider[KnowledgeProviderConfig, Dict[str, Any]]
):
    """Expose the tolerance/fits knowledge module through ``BaseProvider``."""

    def __init__(self, config: KnowledgeProviderConfig):
        super().__init__(config)

    async def _health_check_impl(self) -> bool:
        try:
            from src.core.knowledge.tolerance import get_limit_deviations, get_tolerance_value

            # Cheap deterministic probes.
            it = get_tolerance_value(10.0, "IT7")
            if it is None:
                return False
            deviations = get_limit_deviations("H", 7, 10.0)
            return deviations is not None
        except Exception:
            return False

    async def _process_impl(self, request: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = request
        _ = kwargs
        from src.core.knowledge.tolerance import COMMON_FITS, TOLERANCE_GRADES
        from src.core.knowledge.tolerance.it_grades import SIZE_RANGES

        return {
            "status": "ok",
            "counts": {
                "common_fits": len(COMMON_FITS),
                "size_ranges": len(SIZE_RANGES),
                "tolerance_grade_tables": len(TOLERANCE_GRADES),
            },
            "examples": {
                "it": {"diameter_mm": 10.0, "grade": "IT7"},
                "fit": {"fit_code": "H7/g6", "diameter_mm": 10.0},
            },
        }


class StandardsKnowledgeProviderAdapter(
    BaseProvider[KnowledgeProviderConfig, Dict[str, Any]]
):
    """Expose the standards library knowledge module through ``BaseProvider``."""

    def __init__(self, config: KnowledgeProviderConfig):
        super().__init__(config)

    async def _health_check_impl(self) -> bool:
        try:
            from src.core.knowledge.standards import get_thread_spec

            spec = get_thread_spec("M10")
            return spec is not None
        except Exception:
            return False

    async def _process_impl(self, request: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = request
        _ = kwargs
        from src.core.knowledge.standards import BEARING_DATABASE, METRIC_THREADS, ORING_DATABASE

        return {
            "status": "ok",
            "counts": {
                "threads": len(METRIC_THREADS),
                "bearings": len(BEARING_DATABASE),
                "orings": len(ORING_DATABASE),
            },
            "examples": {
                "thread": "M10",
                "bearing": "6205",
                "oring": "20x3",
            },
        }


def bootstrap_core_knowledge_providers() -> None:
    """Register built-in knowledge providers in ``ProviderRegistry``."""

    if not ProviderRegistry.exists("knowledge", "tolerance"):

        @ProviderRegistry.register("knowledge", "tolerance")
        class ToleranceCoreProvider(ToleranceKnowledgeProviderAdapter):
            def __init__(self, config: Optional[KnowledgeProviderConfig] = None):
                cfg = config or KnowledgeProviderConfig(
                    name="tolerance",
                    provider_type="knowledge",
                    provider_name="tolerance",
                )
                super().__init__(config=cfg)

    if not ProviderRegistry.exists("knowledge", "standards"):

        @ProviderRegistry.register("knowledge", "standards")
        class StandardsCoreProvider(StandardsKnowledgeProviderAdapter):
            def __init__(self, config: Optional[KnowledgeProviderConfig] = None):
                cfg = config or KnowledgeProviderConfig(
                    name="standards",
                    provider_type="knowledge",
                    provider_name="standards",
                )
                super().__init__(config=cfg)
