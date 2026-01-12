"""Drawing recognition endpoint (title block + key fields)."""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from src.api.v1.ocr import get_manager
from src.core.errors import ErrorCode
from src.core.ocr.base import TitleBlock
from src.core.ocr.exceptions import OcrError
from src.middleware.rate_limit import rate_limit
from src.security.input_validator import validate_and_read
from src.utils.idempotency import check_idempotency, store_idempotency

logger = logging.getLogger(__name__)
router = APIRouter(tags=["drawing"])

FIELD_LABELS: Dict[str, str] = {
    "drawing_number": "Drawing Number",
    "revision": "Revision",
    "part_name": "Part Name",
    "material": "Material",
    "scale": "Scale",
    "sheet": "Sheet",
    "date": "Date",
    "weight": "Weight",
    "company": "Company",
    "projection": "Projection",
}


class DrawingField(BaseModel):
    key: str
    label: str
    value: Optional[str] = None
    confidence: Optional[float] = None


class DrawingRecognitionResponse(BaseModel):
    success: bool = Field(True, description="Whether recognition succeeded")
    provider: str
    confidence: Optional[float] = None
    processing_time_ms: Optional[int] = None
    fields: List[DrawingField] = Field(default_factory=list)
    dimensions: List[Dict] = Field(default_factory=list)
    symbols: List[Dict] = Field(default_factory=list)
    error: Optional[str] = None
    code: Optional[ErrorCode] = None


def _build_fields(
    title_block: TitleBlock,
    confidence: Optional[float],
    field_confidence: Optional[Dict[str, float]] = None,
) -> List[DrawingField]:
    fields: List[DrawingField] = []
    for key, label in FIELD_LABELS.items():
        value = getattr(title_block, key, None)
        field_score = None
        if value is not None:
            if field_confidence and key in field_confidence:
                field_score = field_confidence[key]
            else:
                field_score = confidence
        fields.append(
            DrawingField(
                key=key,
                label=label,
                value=value,
                confidence=field_score,
            )
        )
    return fields


def _input_error_response(provider: str, detail: str) -> DrawingRecognitionResponse:
    from src.utils.metrics import ocr_errors_total, ocr_input_rejected_total

    detail_lower = detail.lower()
    if "mime" in detail_lower:
        reason = "invalid_mime"
    elif "too large" in detail_lower:
        reason = "file_too_large"
    elif "page count" in detail_lower:
        reason = "pdf_pages_exceed"
    elif "forbidden token" in detail_lower:
        reason = "pdf_forbidden_token"
    else:
        reason = "validation_failed"
    ocr_input_rejected_total.labels(reason=reason).inc()
    ocr_errors_total.labels(
        provider=provider,
        code=ErrorCode.INPUT_ERROR.value,
        stage="preprocess",
    ).inc()
    return DrawingRecognitionResponse(
        success=False,
        provider=provider,
        confidence=None,
        processing_time_ms=0,
        fields=[],
        dimensions=[],
        symbols=[],
        error=detail,
        code=ErrorCode.INPUT_ERROR,
    )


@router.post("/recognize", response_model=DrawingRecognitionResponse)
async def recognize_drawing(
    file: UploadFile = File(...),
    provider: str = "auto",
    request: Request = None,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
) -> DrawingRecognitionResponse:
    trace_id = str(uuid.uuid4())

    if idempotency_key:
        cached_response = await check_idempotency(idempotency_key, endpoint="drawing")
        if cached_response:
            logger.info(
                "drawing.recognize.idempotency_hit",
                extra={
                    "idempotency_key": idempotency_key,
                    "trace_id": trace_id,
                },
            )
            return DrawingRecognitionResponse(**cached_response)

    if request is not None:
        rate_limit(request)

    try:
        image_bytes, _ = await validate_and_read(file)
    except HTTPException as ve:
        return _input_error_response(provider, str(ve.detail))
    except Exception:
        return _input_error_response(provider, "Input validation failed")

    manager = get_manager()
    try:
        result = await manager.extract(
            image_bytes,
            strategy=provider,
            trace_id=trace_id,
        )
    except OcrError as oe:
        return DrawingRecognitionResponse(
            success=False,
            provider=provider,
            confidence=None,
            processing_time_ms=0,
            fields=[],
            dimensions=[],
            symbols=[],
            error=str(oe),
            code=oe.code if isinstance(oe.code, ErrorCode) else ErrorCode.INTERNAL_ERROR,
        )
    except Exception:
        from src.utils.metrics import ocr_errors_total

        ocr_errors_total.labels(
            provider=provider,
            code=ErrorCode.INTERNAL_ERROR.value,
            stage="infer",
        ).inc()
        return DrawingRecognitionResponse(
            success=False,
            provider=provider,
            confidence=None,
            processing_time_ms=0,
            fields=[],
            dimensions=[],
            symbols=[],
            error="Drawing recognition failed",
            code=ErrorCode.INTERNAL_ERROR,
        )

    confidence = result.calibrated_confidence or result.confidence
    response = DrawingRecognitionResponse(
        provider=result.provider or provider,
        confidence=confidence,
        processing_time_ms=result.processing_time_ms,
        fields=_build_fields(result.title_block, confidence, result.title_block_confidence),
        dimensions=[d.model_dump() for d in result.dimensions],
        symbols=[s.model_dump() for s in result.symbols],
    )

    logger.info(
        "drawing.recognize",
        extra={
            "provider": result.provider or provider,
            "latency_ms": result.processing_time_ms,
            "fields_count": len(response.fields),
            "dimensions_count": len(response.dimensions),
            "symbols_count": len(response.symbols),
            "confidence": confidence,
            "trace_id": trace_id,
            "idempotency_key": idempotency_key,
        },
    )

    if idempotency_key:
        await store_idempotency(
            idempotency_key,
            response.model_dump(),
            endpoint="drawing",
        )

    return response
