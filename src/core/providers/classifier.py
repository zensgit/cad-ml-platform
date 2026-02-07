"""Classifier bridge for the core provider framework.

This module adapts ML classifiers (Hybrid/Graph2D) into the generic
``src.core.providers`` abstractions. Keep imports lazy so environments without
optional ML dependencies (e.g. torch) don't fail at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.core.providers.base import BaseProvider, ProviderConfig
from src.core.providers.registry import ProviderRegistry


@dataclass
class ClassifierProviderConfig(ProviderConfig):
    """Configuration for classifier adapter providers."""

    provider_name: str = "hybrid"
    provider_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassifierRequest:
    """Common request container for classifier providers.

    Not every classifier requires every field:
    - filename: used by Hybrid / Filename rules
    - file_bytes: used by Graph2D / TitleBlock / Process text extraction
    """

    filename: str
    file_bytes: Optional[bytes] = None


class HybridClassifierProviderAdapter(
    BaseProvider[ClassifierProviderConfig, Dict[str, Any]]
):
    """Adapter that exposes ``HybridClassifier`` through ``BaseProvider``."""

    def __init__(
        self, config: ClassifierProviderConfig, wrapped_classifier: Any = None
    ):
        super().__init__(config)
        self._wrapped_classifier = (
            wrapped_classifier or self._build_default_classifier()
        )

    @staticmethod
    def _build_default_classifier() -> Any:
        from src.ml.hybrid_classifier import get_hybrid_classifier

        return get_hybrid_classifier()

    async def _process_impl(self, request: Any, **kwargs: Any) -> Dict[str, Any]:
        if not isinstance(request, ClassifierRequest):
            raise TypeError(
                "HybridClassifierProviderAdapter expects ClassifierRequest as request"
            )
        if not request.filename:
            raise ValueError("filename cannot be empty")
        result = self._wrapped_classifier.classify(
            filename=request.filename,
            file_bytes=request.file_bytes,
            graph2d_result=kwargs.get("graph2d_result"),
        )
        return result.to_dict()

    async def _health_check_impl(self) -> bool:
        # Cheap probe: filename-only path does not require optional ML deps.
        try:
            probe = self._wrapped_classifier.classify(filename="provider-health.dxf")
            return bool(getattr(probe, "to_dict", None))
        except Exception:
            return False


class Graph2DClassifierProviderAdapter(
    BaseProvider[ClassifierProviderConfig, Dict[str, Any]]
):
    """Adapter that exposes ``Graph2DClassifier`` through ``BaseProvider``."""

    def __init__(
        self,
        config: ClassifierProviderConfig,
        wrapped_classifier: Any = None,
        ensemble: bool = False,
    ):
        super().__init__(config)
        self._ensemble = bool(ensemble)
        self._wrapped_classifier = (
            wrapped_classifier or self._build_default_classifier()
        )

    def _build_default_classifier(self) -> Any:
        from src.ml.vision_2d import get_2d_classifier, get_ensemble_2d_classifier

        return get_ensemble_2d_classifier() if self._ensemble else get_2d_classifier()

    async def _process_impl(self, request: Any, **kwargs: Any) -> Dict[str, Any]:
        if not isinstance(request, ClassifierRequest):
            raise TypeError(
                "Graph2DClassifierProviderAdapter expects ClassifierRequest as request"
            )
        if not request.file_bytes:
            raise ValueError("file_bytes cannot be empty for graph2d classification")
        if not request.filename:
            raise ValueError("filename cannot be empty")
        payload = self._wrapped_classifier.predict_from_bytes(
            request.file_bytes, request.filename
        )
        # Add a tiny bit of context for callers using the provider registry.
        if isinstance(payload, dict):
            payload.setdefault("ensemble_enabled", self._ensemble)
        return payload

    async def _health_check_impl(self) -> bool:
        # Don't parse DXF for health. Instead, rely on model loaded flags when present.
        loaded = bool(getattr(self._wrapped_classifier, "_loaded", False))
        return loaded


def bootstrap_core_classifier_providers() -> None:
    """Register built-in classifier providers in ``ProviderRegistry``.

    Registers:
    - ``classifier/hybrid``
    - ``classifier/graph2d``
    - ``classifier/graph2d_ensemble``
    """

    if not ProviderRegistry.exists("classifier", "hybrid"):

        @ProviderRegistry.register("classifier", "hybrid")
        class HybridCoreProvider(HybridClassifierProviderAdapter):
            def __init__(self, config: Optional[ClassifierProviderConfig] = None):
                cfg = config or ClassifierProviderConfig(
                    name="hybrid",
                    provider_type="classifier",
                    provider_name="hybrid",
                )
                super().__init__(config=cfg)

    if not ProviderRegistry.exists("classifier", "graph2d"):

        @ProviderRegistry.register("classifier", "graph2d")
        class Graph2DCoreProvider(Graph2DClassifierProviderAdapter):
            def __init__(self, config: Optional[ClassifierProviderConfig] = None):
                cfg = config or ClassifierProviderConfig(
                    name="graph2d",
                    provider_type="classifier",
                    provider_name="graph2d",
                )
                super().__init__(config=cfg, ensemble=False)

    if not ProviderRegistry.exists("classifier", "graph2d_ensemble"):

        @ProviderRegistry.register("classifier", "graph2d_ensemble")
        class Graph2DEnsembleCoreProvider(Graph2DClassifierProviderAdapter):
            def __init__(self, config: Optional[ClassifierProviderConfig] = None):
                cfg = config or ClassifierProviderConfig(
                    name="graph2d_ensemble",
                    provider_type="classifier",
                    provider_name="graph2d_ensemble",
                )
                super().__init__(config=cfg, ensemble=True)
