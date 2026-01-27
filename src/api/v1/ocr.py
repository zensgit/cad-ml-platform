"""OCR extraction endpoint (scaffold)."""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from src.core.assembly.confidence_calibrator import ConfidenceCalibrationSystem
from src.core.errors import ErrorCode
from src.core.ocr.exceptions import OcrError
from src.core.ocr.manager import OcrManager
from src.core.ocr.providers.deepseek_hf import DeepSeekHfProvider
from src.core.ocr.providers.paddle import PaddleOcrProvider
from src.middleware.rate_limit import rate_limit
from src.security.input_validator import validate_and_read, validate_bytes
from src.utils.idempotency import check_idempotency, store_idempotency

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ocr"])


# Initialize manager (simple singleton for now)
_manager: Optional[OcrManager] = None
_calibrator = ConfidenceCalibrationSystem(method="isotonic")


def get_manager() -> OcrManager:
    global _manager
    if _manager is None:
        _manager = OcrManager(confidence_fallback=0.85)
        _manager.register_provider("paddle", PaddleOcrProvider())
        _manager.register_provider("deepseek_hf", DeepSeekHfProvider())
    return _manager


class MaterialInfoBrief(BaseModel):
    """材料信息简要"""
    found: bool = Field(..., description="是否在数据库中找到")
    grade: Optional[str] = Field(None, description="标准牌号")
    name: Optional[str] = Field(None, description="材料名称")
    category: Optional[str] = Field(None, description="材料类别")
    group: Optional[str] = Field(None, description="材料组")
    warnings: List[str] = Field(default_factory=list, description="工艺警告")
    recommendations: List[str] = Field(default_factory=list, description="工艺建议")


class OcrResponse(BaseModel):
    success: bool = Field(True, description="Whether OCR succeeded")
    provider: str
    confidence: Optional[float] = None
    fallback_level: Optional[str] = None
    processing_time_ms: Optional[int] = None
    dimensions: List
    symbols: List
    title_block: Dict
    material: Optional[str] = Field(None, description="Extracted material from title block")
    material_info: Optional[MaterialInfoBrief] = Field(None, description="Detailed material classification info")
    process_requirements: Optional[Dict] = Field(None, description="Extracted manufacturing process requirements")
    process_route: Optional[Dict] = Field(None, description="Recommended manufacturing process route")
    error: Optional[str] = None
    code: Optional[ErrorCode] = None


class OcrBase64Request(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image or PDF bytes")
    provider: str = Field("auto", description="OCR provider override")
    filename: Optional[str] = Field(None, description="Optional filename hint")
    content_type: Optional[str] = Field(None, description="Optional MIME hint")


class OcrProviderStatus(BaseModel):
    name: str
    ready: bool
    error: Optional[str] = None


class OcrProvidersResponse(BaseModel):
    providers: List[str] = Field(default_factory=list)
    default: str


class OcrHealthResponse(BaseModel):
    status: str
    providers: List[OcrProviderStatus] = Field(default_factory=list)


def list_provider_names(manager: OcrManager) -> List[str]:
    return list(manager.providers.keys())


def get_default_provider_name(manager: OcrManager) -> str:
    providers = list_provider_names(manager)
    if "paddle" in providers:
        return "paddle"
    if "deepseek_hf" in providers:
        return "deepseek_hf"
    return providers[0] if providers else "unknown"


async def collect_provider_statuses(manager: OcrManager) -> List[OcrProviderStatus]:
    async def _check(name: str, provider: object) -> OcrProviderStatus:
        try:
            ready = await provider.health_check()  # type: ignore[attr-defined]
            return OcrProviderStatus(name=name, ready=bool(ready))
        except Exception as exc:
            return OcrProviderStatus(name=name, ready=False, error=str(exc))

    tasks = [_check(name, provider) for name, provider in manager.providers.items()]
    if not tasks:
        return []
    return list(await asyncio.gather(*tasks))


def summarize_provider_health(statuses: List[OcrProviderStatus]) -> str:
    if not statuses:
        return "unavailable"
    ready_count = sum(1 for status in statuses if status.ready)
    if ready_count == len(statuses):
        return "healthy"
    if ready_count > 0:
        return "degraded"
    return "unhealthy"


def _input_error_response(provider: str, detail: str) -> OcrResponse:
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
    return OcrResponse(
        success=False,
        provider=provider,
        confidence=None,
        fallback_level=None,
        processing_time_ms=0,
        dimensions=[],
        symbols=[],
        title_block={},
        material=None,
        material_info=None,
        process_requirements=None,
        process_route=None,
        error=detail,
        code=ErrorCode.INPUT_ERROR,
    )


def _strip_base64_prefix(payload: str) -> str:
    cleaned = payload.strip()
    if cleaned.startswith("data:") and "base64," in cleaned:
        return cleaned.split("base64,", 1)[1]
    return cleaned


async def _run_ocr_extract(image_bytes: bytes, provider: str, trace_id: str) -> OcrResponse:
    manager = get_manager()
    try:
        result = await manager.extract(
            image_bytes,
            strategy=provider,
            trace_id=trace_id,
        )
    except OcrError as oe:  # provider down, rate limit, circuit, etc.
        return OcrResponse(
            success=False,
            provider=provider,
            confidence=None,
            fallback_level=None,
            processing_time_ms=0,
            dimensions=[],
            symbols=[],
            title_block={},
            material=None,
            material_info=None,
            process_requirements=None,
            process_route=None,
            error=str(oe),
            code=oe.code if isinstance(oe.code, ErrorCode) else ErrorCode.INTERNAL_ERROR,
        )
    except Exception:  # unknown internal error
        from src.utils.metrics import ocr_errors_total

        ocr_errors_total.labels(
            provider=provider,
            code=ErrorCode.INTERNAL_ERROR.value,
            stage="infer",
        ).inc()
        return OcrResponse(
            success=False,
            provider=provider,
            confidence=None,
            fallback_level=None,
            processing_time_ms=0,
            dimensions=[],
            symbols=[],
            title_block={},
            material=None,
            material_info=None,
            process_requirements=None,
            process_route=None,
            error="OCR extraction failed",
            code=ErrorCode.INTERNAL_ERROR,
        )

    # Calibrate confidence (single evidence source for now)
    if result.confidence is not None:
        if _calibrator.calibrator is not None:
            cal = _calibrator.calibrator.calibrate(result.confidence)
            result.calibrated_confidence = cal
        else:
            result.calibrated_confidence = result.confidence
    logger.info(
        "ocr.extract",
        extra={
            "provider": result.provider or provider,
            "latency_ms": result.processing_time_ms,
            "fallback_level": result.fallback_level,
            "image_hash": result.image_hash,
            "completeness": result.completeness,
            "calibrated_confidence": (result.calibrated_confidence or result.confidence),
            "trace_id": trace_id,
            "extraction_mode": result.extraction_mode,
            "dimensions_count": len(result.dimensions),
            "symbols_count": len(result.symbols),
            "stages_latency_ms": result.stages_latency_ms,
        },
    )
    # Generate process route if process requirements exist
    process_route = None
    material = result.title_block.material
    material_info = None

    # Get detailed material info if material is available
    if material:
        try:
            from src.core.materials import classify_material_detailed
            info = classify_material_detailed(material)
            if info:
                material_info = MaterialInfoBrief(
                    found=True,
                    grade=info.grade,
                    name=info.name,
                    category=info.category.value,
                    group=info.group.value,
                    warnings=info.process.warnings,
                    recommendations=info.process.recommendations,
                )
            else:
                material_info = MaterialInfoBrief(found=False)
        except Exception as e:
            logger.warning("material_info.classification_failed", extra={"error": str(e)})

    if result.process_requirements:
        try:
            from src.core.process import generate_process_route
            route = generate_process_route(result.process_requirements, material=material)
            process_route = route.to_dict()
        except Exception as e:
            logger.warning("process_route.generation_failed", extra={"error": str(e)})

    return OcrResponse(
        provider=result.provider or provider,
        confidence=(result.calibrated_confidence or result.confidence),
        fallback_level=result.fallback_level,
        processing_time_ms=result.processing_time_ms,
        dimensions=[d.model_dump() for d in result.dimensions],
        symbols=[s.model_dump() for s in result.symbols],
        title_block=result.title_block.model_dump(),
        material=result.title_block.material,
        material_info=material_info,
        process_requirements=result.process_requirements.model_dump() if result.process_requirements else None,
        process_route=process_route,
    )


@router.post("/extract", response_model=OcrResponse)
async def ocr_extract(
    file: UploadFile = File(...),
    provider: str = "auto",
    request: Request = None,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    trace_id = str(uuid.uuid4())

    # Check idempotency cache first
    if idempotency_key:
        cached_response = await check_idempotency(
            idempotency_key,
            endpoint="ocr",
        )
        if cached_response:
            logger.info(
                "ocr.extract.idempotency_hit",
                extra={
                    "idempotency_key": idempotency_key,
                    "trace_id": trace_id,
                },
            )
            return OcrResponse(**cached_response)

    if request is not None:
        rate_limit(request)
    try:
        try:
            image_bytes, _ = await validate_and_read(file)
        except HTTPException as ve:
            return _input_error_response(provider, str(ve.detail))
        except Exception:
            return _input_error_response(provider, "Input validation failed")

        response = await _run_ocr_extract(image_bytes, provider, trace_id)

        if idempotency_key and response.success:
            await store_idempotency(
                idempotency_key,
                response.model_dump(),
                endpoint="ocr",
            )

        return response
    except HTTPException as ve:
        from src.utils.metrics import ocr_errors_total, ocr_input_rejected_total

        ocr_input_rejected_total.labels(reason="validation_failed").inc()
        ocr_errors_total.labels(
            provider=provider,
            code=ErrorCode.INPUT_ERROR.value,
            stage="preprocess",
        ).inc()
        return OcrResponse(
            success=False,
            provider=provider,
            confidence=None,
            fallback_level=None,
            processing_time_ms=0,
            dimensions=[],
            symbols=[],
            title_block={},
            material=None,
            material_info=None,
            process_requirements=None,
            process_route=None,
            error=str(ve.detail),
            code=ErrorCode.INPUT_ERROR,
        )
    except Exception:  # noqa
        logger.error(
            "ocr.extract_failed",
            extra={"trace_id": trace_id},
        )
        from src.utils.metrics import ocr_errors_total

        ocr_errors_total.labels(
            provider=provider,
            code=ErrorCode.INTERNAL_ERROR.value,
            stage="postprocess",
        ).inc()
        return OcrResponse(
            success=False,
            provider=provider,
            confidence=None,
            fallback_level=None,
            processing_time_ms=0,
            dimensions=[],
            symbols=[],
            title_block={},
            material=None,
            material_info=None,
            process_requirements=None,
            process_route=None,
            error="OCR extraction failed",
            code=ErrorCode.INTERNAL_ERROR,
        )


@router.post("/extract-base64", response_model=OcrResponse)
async def ocr_extract_base64(
    payload: OcrBase64Request,
    request: Request = None,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
) -> OcrResponse:
    trace_id = str(uuid.uuid4())
    provider = payload.provider

    if idempotency_key:
        cached_response = await check_idempotency(
            idempotency_key,
            endpoint="ocr",
        )
        if cached_response:
            logger.info(
                "ocr.extract.idempotency_hit",
                extra={
                    "idempotency_key": idempotency_key,
                    "trace_id": trace_id,
                },
            )
            return OcrResponse(**cached_response)

    if request is not None:
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

    response = await _run_ocr_extract(image_bytes, provider, trace_id)

    if idempotency_key and response.success:
        await store_idempotency(
            idempotency_key,
            response.model_dump(),
            endpoint="ocr",
        )

    return response


@router.get("/providers", response_model=OcrProvidersResponse)
async def list_ocr_providers() -> OcrProvidersResponse:
    manager = get_manager()
    providers = list_provider_names(manager)
    return OcrProvidersResponse(providers=providers, default=get_default_provider_name(manager))


@router.get("/health", response_model=OcrHealthResponse)
async def ocr_health() -> OcrHealthResponse:
    manager = get_manager()
    statuses = await collect_provider_statuses(manager)
    return OcrHealthResponse(status=summarize_provider_health(statuses), providers=statuses)
