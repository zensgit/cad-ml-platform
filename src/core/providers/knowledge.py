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


class DesignStandardsKnowledgeProviderAdapter(
    BaseProvider[KnowledgeProviderConfig, Dict[str, Any]]
):
    """Expose the design-standards knowledge module through ``BaseProvider``."""

    def __init__(self, config: KnowledgeProviderConfig):
        super().__init__(config)

    async def _health_check_impl(self) -> bool:
        try:
            from src.core.knowledge.design_standards import (
                GeneralToleranceClass,
                SurfaceFinishGrade,
                get_linear_tolerance,
                get_preferred_diameter,
                get_ra_value,
            )

            ra = get_ra_value(SurfaceFinishGrade.N7)
            if ra <= 0:
                return False

            tol = get_linear_tolerance(50.0, GeneralToleranceClass.M)
            if tol is None or tol <= 0:
                return False

            diameter = get_preferred_diameter(23.0)
            return diameter is not None and diameter > 0
        except Exception:
            return False

    async def _process_impl(self, request: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = request
        _ = kwargs
        from src.core.knowledge.design_standards import (
            LINEAR_TOLERANCE_TABLE,
            PREFERRED_DIAMETERS,
            STANDARD_CHAMFERS,
            STANDARD_FILLETS,
            SURFACE_FINISH_TABLE,
        )

        return {
            "status": "ok",
            "counts": {
                "surface_finish_grades": len(SURFACE_FINISH_TABLE),
                "linear_tolerance_ranges": len(LINEAR_TOLERANCE_TABLE),
                "preferred_diameters": len(PREFERRED_DIAMETERS),
                "standard_chamfers": len(STANDARD_CHAMFERS),
                "standard_fillets": len(STANDARD_FILLETS),
            },
            "examples": {
                "surface_finish_grade": "N7",
                "surface_finish_application": "bearing_journal",
                "linear_tolerance": {"dimension_mm": 50.0, "tolerance_class": "m"},
                "preferred_diameter": {"target_mm": 23.0, "direction": "nearest"},
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

    if not ProviderRegistry.exists("knowledge", "design_standards"):

        @ProviderRegistry.register("knowledge", "design_standards")
        class DesignStandardsCoreProvider(DesignStandardsKnowledgeProviderAdapter):
            def __init__(self, config: Optional[KnowledgeProviderConfig] = None):
                cfg = config or KnowledgeProviderConfig(
                    name="design_standards",
                    provider_type="knowledge",
                    provider_name="design_standards",
                )
                super().__init__(config=cfg)
