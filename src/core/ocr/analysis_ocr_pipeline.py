"""Shared OCR enrichment helper for analyze flows."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from .manager import OcrManager

ProviderGetter = Callable[[str, str], Any]
BootstrapFn = Callable[[], Any]
ManagerFactory = Callable[[], OcrManager]


def _default_bootstrap() -> Any:
    from src.core.providers import bootstrap_core_provider_registry

    return bootstrap_core_provider_registry()


def _default_provider_getter(kind: str, name: str) -> Any:
    from src.core.providers import ProviderRegistry

    return ProviderRegistry.get(kind, name)


def _default_manager_factory() -> OcrManager:
    return OcrManager(confidence_fallback=0.85)


async def run_analysis_ocr_pipeline(
    *,
    enable_ocr: bool,
    ocr_provider_strategy: str,
    unified_data: Mapping[str, Any],
    bootstrap_registry_fn: Optional[BootstrapFn] = None,
    provider_getter: Optional[ProviderGetter] = None,
    manager_factory: Optional[ManagerFactory] = None,
) -> Optional[Dict[str, Any]]:
    if not enable_ocr:
        return None

    (bootstrap_registry_fn or _default_bootstrap)()
    getter = provider_getter or _default_provider_getter
    ocr_manager = (manager_factory or _default_manager_factory)()
    for provider_name in ("paddle", "deepseek_hf"):
        provider = getter("ocr", provider_name)
        if provider is not None:
            ocr_manager.register_provider(provider_name, provider)

    img_bytes = unified_data.get("preview_image_bytes")
    if not img_bytes:
        return {"status": "no_preview_image"}

    ocr_result = await ocr_manager.extract(img_bytes, strategy=ocr_provider_strategy)
    return {
        "provider": ocr_result.provider,
        "confidence": ocr_result.calibrated_confidence or ocr_result.confidence,
        "fallback_level": ocr_result.fallback_level,
        "dimensions": [item.model_dump() for item in ocr_result.dimensions],
        "symbols": [item.model_dump() for item in ocr_result.symbols],
        "completeness": ocr_result.completeness,
    }


__all__ = ["run_analysis_ocr_pipeline"]
