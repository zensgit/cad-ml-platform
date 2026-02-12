"""Classifier bridge for the core provider framework.

This module adapts ML classifiers (Hybrid/Graph2D) into the generic
``src.core.providers`` abstractions. Keep imports lazy so environments without
optional ML dependencies (e.g. torch) don't fail at import time.
"""

from __future__ import annotations

import importlib.util
import os
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
    file_path: Optional[str] = None


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
        # Keep health checks cheap and deterministic:
        # - respect feature flag
        # - avoid parsing DXF
        # - avoid forcing model load; instead validate prerequisites (torch + model files)
        if os.getenv("GRAPH2D_ENABLED", "false").lower() != "true":
            raise RuntimeError("disabled_by_config")
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("torch_missing")

        if self._ensemble:
            env_paths = os.getenv("GRAPH2D_ENSEMBLE_MODELS", "").strip()
            model_paths = (
                [p.strip() for p in env_paths.split(",") if p.strip()]
                if env_paths
                else [
                    "models/graph2d_edge_sage_v3.pth",
                    "models/graph2d_edge_sage_v4_best.pth",
                ]
            )
            if not any(os.path.exists(path) for path in model_paths):
                raise RuntimeError(f"model_missing: {model_paths}")
            return True

        model_path = os.getenv(
            "GRAPH2D_MODEL_PATH",
            "models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth",
        )
        if not os.path.exists(model_path):
            raise RuntimeError(f"model_missing: {model_path}")
        return True


class V16PartClassifierProviderAdapter(
    BaseProvider[ClassifierProviderConfig, Dict[str, Any]]
):
    """Adapter that exposes the V16 part classifier through ``BaseProvider``.

    Note: V16 lives in ``src/ml/part_classifier.py`` which imports torch.
    This adapter keeps imports lazy and does not load the model for health checks.
    """

    def __init__(self, config: ClassifierProviderConfig):
        super().__init__(config)

    @staticmethod
    def _has_torch() -> bool:
        return importlib.util.find_spec("torch") is not None

    @staticmethod
    def _models_present() -> bool:
        v6_path = os.getenv("CAD_CLASSIFIER_MODEL", "models/cad_classifier_v6.pt")
        v14_path = "models/cad_classifier_v14_ensemble.pt"
        return os.path.exists(v6_path) and os.path.exists(v14_path)

    async def _health_check_impl(self) -> bool:
        if os.getenv("DISABLE_V16_CLASSIFIER", "").lower() in ("1", "true", "yes"):
            raise RuntimeError("disabled_by_config")
        if not self._has_torch():
            raise RuntimeError("torch_missing")
        if not self._models_present():
            v6_path = os.getenv("CAD_CLASSIFIER_MODEL", "models/cad_classifier_v6.pt")
            v14_path = "models/cad_classifier_v14_ensemble.pt"
            raise RuntimeError(f"model_missing: v6={v6_path} v14={v14_path}")
        return True

    async def _process_impl(self, request: Any, **kwargs: Any) -> Dict[str, Any]:
        if not isinstance(request, ClassifierRequest):
            raise TypeError(
                "V16PartClassifierProviderAdapter expects ClassifierRequest as request"
            )
        if not request.file_path:
            raise ValueError("file_path is required for v16 classification")
        try:
            from src.core.analyzer import _get_v16_classifier

            clf = _get_v16_classifier()
            if clf is None:
                return {"status": "unavailable"}
            result = clf.predict(str(request.file_path))
            if result is None:
                return {"status": "no_prediction"}
            payload: Dict[str, Any] = {
                "status": "ok",
                "label": result.category,
                "confidence": float(result.confidence),
                "probabilities": dict(result.probabilities),
                "model_version": getattr(result, "model_version", "v16"),
                "needs_review": bool(getattr(result, "needs_review", False)),
                "review_reason": getattr(result, "review_reason", None),
                "top2_category": getattr(result, "top2_category", None),
                "top2_confidence": getattr(result, "top2_confidence", None),
            }
            return payload
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": str(exc)}


class V6PartClassifierProviderAdapter(
    BaseProvider[ClassifierProviderConfig, Dict[str, Any]]
):
    """Adapter that exposes the V6 part classifier through ``BaseProvider``."""

    def __init__(self, config: ClassifierProviderConfig):
        super().__init__(config)

    @staticmethod
    def _has_torch() -> bool:
        return importlib.util.find_spec("torch") is not None

    @staticmethod
    def _model_present() -> bool:
        v6_path = os.getenv("CAD_CLASSIFIER_MODEL", "models/cad_classifier_v6.pt")
        return os.path.exists(v6_path)

    async def _health_check_impl(self) -> bool:
        if not self._has_torch():
            raise RuntimeError("torch_missing")
        if not self._model_present():
            v6_path = os.getenv("CAD_CLASSIFIER_MODEL", "models/cad_classifier_v6.pt")
            raise RuntimeError(f"model_missing: {v6_path}")
        return True

    async def _process_impl(self, request: Any, **kwargs: Any) -> Dict[str, Any]:
        if not isinstance(request, ClassifierRequest):
            raise TypeError(
                "V6PartClassifierProviderAdapter expects ClassifierRequest as request"
            )
        if not request.file_path:
            raise ValueError("file_path is required for v6 classification")
        try:
            from src.core.analyzer import _get_ml_classifier

            clf = _get_ml_classifier()
            if clf is None:
                return {"status": "unavailable"}
            result = clf.predict(str(request.file_path))
            if result is None:
                return {"status": "no_prediction"}
            return {
                "status": "ok",
                "label": result.category,
                "confidence": float(result.confidence),
                "probabilities": dict(result.probabilities),
                "model_version": getattr(result, "model_version", "v6"),
            }
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": str(exc)}


def bootstrap_core_classifier_providers() -> None:
    """Register built-in classifier providers in ``ProviderRegistry``.

    Registers:
    - ``classifier/hybrid``
    - ``classifier/graph2d``
    - ``classifier/graph2d_ensemble``
    - ``classifier/v16``
    - ``classifier/v6``
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

    if not ProviderRegistry.exists("classifier", "v16"):

        @ProviderRegistry.register("classifier", "v16")
        class V16CoreProvider(V16PartClassifierProviderAdapter):
            def __init__(self, config: Optional[ClassifierProviderConfig] = None):
                cfg = config or ClassifierProviderConfig(
                    name="v16",
                    provider_type="classifier",
                    provider_name="v16",
                )
                super().__init__(config=cfg)

    if not ProviderRegistry.exists("classifier", "v6"):

        @ProviderRegistry.register("classifier", "v6")
        class V6CoreProvider(V6PartClassifierProviderAdapter):
            def __init__(self, config: Optional[ClassifierProviderConfig] = None):
                cfg = config or ClassifierProviderConfig(
                    name="v6",
                    provider_type="classifier",
                    provider_name="v6",
                )
                super().__init__(config=cfg)
