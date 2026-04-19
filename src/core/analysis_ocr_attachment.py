from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional


OCRPipeline = Callable[..., Awaitable[Optional[Dict[str, Any]]]]


async def attach_analysis_ocr_payload(
    *,
    enable_ocr: bool,
    ocr_provider_strategy: str,
    unified_data: Dict[str, Any],
    results: Dict[str, Any],
    ocr_pipeline: OCRPipeline,
) -> Optional[Dict[str, Any]]:
    ocr_payload = await ocr_pipeline(
        enable_ocr=enable_ocr,
        ocr_provider_strategy=ocr_provider_strategy,
        unified_data=unified_data,
    )
    if ocr_payload is not None:
        results["ocr"] = ocr_payload
    return ocr_payload
