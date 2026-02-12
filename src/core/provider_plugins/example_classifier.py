"""Example provider plugin: classifier/example_rules.

This is a minimal, dependency-free example of how to register a provider via
`CORE_PROVIDER_PLUGINS`. It intentionally does *not* import torch or parse DXF;
it only uses filename heuristics to demonstrate the wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.core.providers.base import BaseProvider, ProviderConfig
from src.core.providers.classifier import ClassifierRequest
from src.core.providers.registry import ProviderRegistry


@dataclass
class ExampleRulesClassifierConfig(ProviderConfig):
    """Config for the example rules-based classifier plugin."""

    default_label: str = "unknown"
    drawing_label: str = "mechanical_drawing"
    min_confidence: float = 0.75


class ExampleRulesClassifierProvider(
    BaseProvider[ExampleRulesClassifierConfig, Dict[str, Any]]
):
    """A tiny rules-based classifier (filename-only)."""

    def __init__(self, config: Optional[ExampleRulesClassifierConfig] = None):
        cfg = config or ExampleRulesClassifierConfig(
            name="example_rules",
            provider_type="classifier",
        )
        super().__init__(cfg)

    def _classify(self, filename: str) -> Tuple[str, float, str]:
        name = (filename or "").strip()
        lower = name.lower()

        # Extremely small heuristic set (demo only).
        if lower.endswith((".dxf", ".dwg")):
            return self.config.drawing_label, float(self.config.min_confidence), "ext"
        if lower.endswith((".step", ".stp", ".iges", ".igs")):
            return "cad_model", 0.6, "ext"
        return self.config.default_label, 0.0, "fallback"

    async def _process_impl(self, request: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        if not isinstance(request, ClassifierRequest):
            raise TypeError(
                "ExampleRulesClassifierProvider expects ClassifierRequest as request"
            )
        if not request.filename:
            raise ValueError("filename cannot be empty")

        label, confidence, reason = self._classify(request.filename)
        return {
            "status": "ok",
            "label": label,
            "confidence": float(confidence),
            "source": "example_rules",
            "reason": reason,
            "provider": "example_rules",
        }

    async def _health_check_impl(self) -> bool:
        # Health is always OK: this provider has no external dependencies.
        return True


def bootstrap() -> None:
    """Plugin entrypoint used by `CORE_PROVIDER_PLUGINS=...:bootstrap`."""
    if ProviderRegistry.exists("classifier", "example_rules"):
        return
    ProviderRegistry.register("classifier", "example_rules")(ExampleRulesClassifierProvider)


__all__ = [
    "ExampleRulesClassifierConfig",
    "ExampleRulesClassifierProvider",
    "bootstrap",
]

