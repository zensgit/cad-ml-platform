"""Drawing recognition endpoint (title block + key fields)."""

from __future__ import annotations

import base64
import binascii
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from src.api.v1.ocr import (
    OcrHealthResponse,
    OcrProvidersResponse,
    collect_provider_statuses,
    get_default_provider_name,
    get_manager,
    list_provider_names,
    summarize_provider_health,
)
from src.core.errors import ErrorCode
from src.core.ocr.base import OcrResult, ProcessRequirements, SymbolType, TitleBlock
from src.core.ocr.exceptions import OcrError
from src.middleware.rate_limit import rate_limit
from src.security.input_validator import validate_and_read, validate_bytes
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


class DrawingRecognitionBase64Request(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image or PDF bytes")
    provider: str = Field("auto", description="OCR provider override")
    filename: Optional[str] = Field(None, description="Optional filename hint")
    content_type: Optional[str] = Field(None, description="Optional MIME hint")


class DrawingFieldDefinition(BaseModel):
    key: str
    label: str


class DrawingFieldCatalogResponse(BaseModel):
    fields: List[DrawingFieldDefinition] = Field(default_factory=list)


class DrawingRecognitionResponse(BaseModel):
    success: bool = Field(True, description="Whether recognition succeeded")
    provider: str
    confidence: Optional[float] = None
    processing_time_ms: Optional[int] = None
    title_block: Dict[str, Optional[str]] = Field(default_factory=dict)
    field_confidence: Dict[str, Optional[float]] = Field(default_factory=dict)
    fields: List[DrawingField] = Field(default_factory=list)
    dimensions: List[Dict[str, Any]] = Field(default_factory=list)
    symbols: List[Dict[str, Any]] = Field(default_factory=list)
    process_requirements: ProcessRequirements = Field(default_factory=ProcessRequirements)
    field_coverage: Dict[str, Any] = Field(default_factory=dict)
    engineering_signals: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    code: Optional[ErrorCode] = None


STANDARD_CANDIDATE_PATTERN = re.compile(
    r"(?:GB/T|GB|ISO|DIN|ANSI|ASME|ASTM|JIS|EN)\s*[-/]?\s*[A-Z0-9.\-]+",
    re.IGNORECASE,
)


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


def _build_field_confidence(
    title_block: TitleBlock,
    confidence: Optional[float],
    field_confidence: Optional[Dict[str, float]] = None,
) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {}
    for key in FIELD_LABELS:
        value = getattr(title_block, key, None)
        if value is None:
            result[key] = None
            continue
        if field_confidence and key in field_confidence:
            result[key] = field_confidence[key]
        else:
            result[key] = confidence
    return result


def _build_field_coverage(title_block: TitleBlock) -> Dict[str, Any]:
    recognized_keys = [key for key in FIELD_LABELS if getattr(title_block, key, None)]
    missing_keys = [key for key in FIELD_LABELS if key not in recognized_keys]
    total_fields = len(FIELD_LABELS)
    recognized_count = len(recognized_keys)
    coverage_ratio = (recognized_count / total_fields) if total_fields else 0.0
    return {
        "recognized_count": recognized_count,
        "total_fields": total_fields,
        "coverage_ratio": round(coverage_ratio, 4),
        "recognized_keys": recognized_keys,
        "missing_keys": missing_keys,
    }


def _extract_standard_candidates(process_requirements: ProcessRequirements) -> List[str]:
    candidates: List[str] = []
    seen = set()

    def _add(value: Optional[str]) -> None:
        if not value:
            return
        normalized = value.strip()
        if not normalized:
            return
        normalized = re.sub(r"\s+", "", normalized)
        upper = normalized.upper()
        if upper in seen:
            return
        seen.add(upper)
        candidates.append(normalized)

    for surface_treatment in process_requirements.surface_treatments:
        _add(surface_treatment.standard)

    notes_blob = "\n".join(process_requirements.general_notes)
    if process_requirements.raw_text:
        notes_blob = f"{notes_blob}\n{process_requirements.raw_text}".strip()
    for match in STANDARD_CANDIDATE_PATTERN.finditer(notes_blob):
        _add(match.group(0).replace(" ", ""))
    return candidates


def _build_engineering_signals(result: OcrResult) -> Dict[str, Any]:
    process_requirements = result.process_requirements
    symbol_types = sorted({symbol.type.value for symbol in result.symbols})
    gdt_symbol_types = sorted(
        {
            symbol.type.value
            for symbol in result.symbols
            if symbol.type != SymbolType.surface_roughness
        }
    )
    materials_detected = []
    if result.title_block.material:
        materials_detected.append(result.title_block.material)

    return {
        "dimension_count": len(result.dimensions),
        "symbol_count": len(result.symbols),
        "symbol_types": symbol_types,
        "gdt_symbol_types": gdt_symbol_types,
        "has_surface_finish": SymbolType.surface_roughness.value in symbol_types,
        "has_gdt": bool(gdt_symbol_types),
        "process_requirement_counts": {
            "heat_treatments": len(process_requirements.heat_treatments),
            "surface_treatments": len(process_requirements.surface_treatments),
            "welding": len(process_requirements.welding),
            "general_notes": len(process_requirements.general_notes),
        },
        "materials_detected": materials_detected,
        "standards_candidates": _extract_standard_candidates(process_requirements),
    }


def _input_error_response(provider: str, detail: str) -> DrawingRecognitionResponse:
    from src.utils.metrics import ocr_errors_total, ocr_input_rejected_total

    detail_lower = detail.lower()
    if "mime" in detail_lower:
        reason = "invalid_mime"
    elif "base64" in detail_lower:
        reason = "base64_invalid"
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
        title_block={},
        field_confidence={},
        fields=[],
        dimensions=[],
        symbols=[],
        error=detail,
        code=ErrorCode.INPUT_ERROR,
    )


def _strip_base64_prefix(payload: str) -> str:
    cleaned = payload.strip()
    if cleaned.startswith("data:") and "base64," in cleaned:
        return cleaned.split("base64,", 1)[1]
    return cleaned


async def _run_recognition(
    image_bytes: bytes,
    provider: str,
    trace_id: str,
    idempotency_key: Optional[str],
) -> DrawingRecognitionResponse:
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
            title_block={},
            field_confidence={},
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
            title_block={},
            field_confidence={},
            fields=[],
            dimensions=[],
            symbols=[],
            error="Drawing recognition failed",
            code=ErrorCode.INTERNAL_ERROR,
        )

    confidence = result.calibrated_confidence or result.confidence
    response = DrawingRecognitionResponse(
        success=True,
        provider=result.provider or provider,
        confidence=confidence,
        processing_time_ms=result.processing_time_ms,
        title_block=result.title_block.model_dump(),
        field_confidence=_build_field_confidence(
            result.title_block,
            confidence,
            result.title_block_confidence,
        ),
        fields=_build_fields(result.title_block, confidence, result.title_block_confidence),
        dimensions=[d.model_dump() for d in result.dimensions],
        symbols=[s.model_dump() for s in result.symbols],
        process_requirements=result.process_requirements,
        field_coverage=_build_field_coverage(result.title_block),
        engineering_signals=_build_engineering_signals(result),
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
    return response


@router.post("/recognize", response_model=DrawingRecognitionResponse)
async def recognize_drawing(
    request: Request,
    file: UploadFile = File(...),
    provider: str = "auto",
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

    rate_limit(request)

    try:
        image_bytes, _ = await validate_and_read(file)
    except HTTPException as ve:
        return _input_error_response(provider, str(ve.detail))
    except Exception:
        return _input_error_response(provider, "Input validation failed")

    response = await _run_recognition(image_bytes, provider, trace_id, idempotency_key)

    if idempotency_key and response.success:
        await store_idempotency(
            idempotency_key,
            response.model_dump(),
            endpoint="drawing",
        )

    return response


@router.post("/recognize-base64", response_model=DrawingRecognitionResponse)
async def recognize_drawing_base64(
    payload: DrawingRecognitionBase64Request,
    request: Request,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
) -> DrawingRecognitionResponse:
    trace_id = str(uuid.uuid4())
    provider = payload.provider

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

    rate_limit(request)

    try:
        cleaned = _strip_base64_prefix(payload.image_base64)
        if not cleaned:
            return _input_error_response(provider, "Base64 payload empty")
        image_bytes = base64.b64decode(cleaned, validate=True)
        if not image_bytes:
            return _input_error_response(provider, "Base64 payload empty")
    except (binascii.Error, ValueError) as exc:
        return _input_error_response(provider, f"Invalid base64 image data: {exc}")

    try:
        image_bytes, _ = validate_bytes(
            image_bytes,
            filename=payload.filename or "",
            content_type=payload.content_type,
        )
    except HTTPException as ve:
        return _input_error_response(provider, str(ve.detail))
    except Exception:
        return _input_error_response(provider, "Input validation failed")

    response = await _run_recognition(image_bytes, provider, trace_id, idempotency_key)

    if idempotency_key and response.success:
        await store_idempotency(
            idempotency_key,
            response.model_dump(),
            endpoint="drawing",
        )

    return response


@router.get("/fields", response_model=DrawingFieldCatalogResponse)
async def list_drawing_fields() -> DrawingFieldCatalogResponse:
    return DrawingFieldCatalogResponse(
        fields=[
            DrawingFieldDefinition(key=key, label=label)
            for key, label in FIELD_LABELS.items()
        ]
    )


@router.get("/providers", response_model=OcrProvidersResponse)
async def list_drawing_providers() -> OcrProvidersResponse:
    manager = get_manager()
    providers = list_provider_names(manager)
    return OcrProvidersResponse(providers=providers, default=get_default_provider_name(manager))


@router.get("/health", response_model=OcrHealthResponse)
async def drawing_health() -> OcrHealthResponse:
    manager = get_manager()
    statuses = await collect_provider_statuses(manager)
    return OcrHealthResponse(status=summarize_provider_health(statuses), providers=statuses)
