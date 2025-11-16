"""OCR extraction endpoint (scaffold)."""

from __future__ import annotations

import logging
import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Header
from pydantic import BaseModel

from src.core.ocr.manager import OcrManager
from src.core.ocr.providers.paddle import PaddleOcrProvider
from src.core.ocr.providers.deepseek_hf import DeepSeekHfProvider
from src.security.input_validator import validate_and_read
from src.core.assembly.confidence_calibrator import ConfidenceCalibrationSystem
from src.utils.logging import setup_logging
from src.middleware.rate_limit import rate_limit
from src.utils.idempotency import check_idempotency, store_idempotency

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/ocr", tags=["ocr"])


# Initialize manager (simple singleton for now)
_manager: OcrManager | None = None
_calibrator = ConfidenceCalibrationSystem(method="isotonic")


def get_manager() -> OcrManager:
    global _manager
    if _manager is None:
        _manager = OcrManager(confidence_fallback=0.85)
        _manager.register_provider("paddle", PaddleOcrProvider())
        _manager.register_provider("deepseek_hf", DeepSeekHfProvider())
    return _manager


class OcrResponse(BaseModel):
    provider: str
    confidence: float | None
    fallback_level: str | None
    processing_time_ms: int | None
    dimensions: list
    symbols: list
    title_block: dict


@router.post("/extract", response_model=OcrResponse)
async def ocr_extract(
    file: UploadFile = File(...),
    provider: str = "auto",
    request: Request = None,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key")
):
    trace_id = str(uuid.uuid4())

    # Check idempotency cache first
    if idempotency_key:
        cached_response = await check_idempotency(idempotency_key, endpoint="ocr")
        if cached_response:
            logger.info(
                "ocr.extract.idempotency_hit",
                extra={
                    "idempotency_key": idempotency_key,
                    "trace_id": trace_id,
                }
            )
            return OcrResponse(**cached_response)

    if request is not None:
        rate_limit(request)
    try:
        image_bytes, mime = await validate_and_read(file)
        manager = get_manager()
        result = await manager.extract(image_bytes, strategy=provider, trace_id=trace_id)
        # Calibrate confidence (single evidence source for now)
        if result.confidence is not None:
            cal = _calibrator.calibrator.calibrate(result.confidence)
            result.calibrated_confidence = cal
        logger.info(
            "ocr.extract",
            extra={
                "provider": result.provider or provider,
                "latency_ms": result.processing_time_ms,
                "fallback_level": result.fallback_level,
                "image_hash": result.image_hash,
                "completeness": result.completeness,
                "calibrated_confidence": result.calibrated_confidence or result.confidence,
                "trace_id": trace_id,
                "extraction_mode": result.extraction_mode,
                "dimensions_count": len(result.dimensions),
                "symbols_count": len(result.symbols),
                "stages_latency_ms": result.stages_latency_ms,
                "idempotency_key": idempotency_key,
            },
        )
        response = OcrResponse(
            provider=result.provider or provider,
            confidence=result.calibrated_confidence or result.confidence,
            fallback_level=result.fallback_level,
            processing_time_ms=result.processing_time_ms,
            dimensions=[d.model_dump() for d in result.dimensions],
            symbols=[s.model_dump() for s in result.symbols],
            title_block=result.title_block.model_dump(),
        )

        # Store in idempotency cache if key provided
        if idempotency_key:
            await store_idempotency(
                idempotency_key,
                response.model_dump(),
                endpoint="ocr"
            )

        return response
    except Exception as e:  # noqa
        logger.error("ocr.extract_failed", extra={"error_code": getattr(e, "code", None), "trace_id": trace_id})
        raise HTTPException(status_code=500, detail="OCR extraction failed")
