"""Shared input validation and document adaptation helpers for analyze flows."""

from __future__ import annotations

import asyncio
import os
import time as _time
from typing import Any, Dict, Optional

from fastapi import HTTPException

from src.core.errors_extended import ErrorCode, build_error
from src.models.cad_document import CadDocument
from src.security.input_validator import (
    deep_format_validate,
    is_supported_mime,
    matrix_validate,
    signature_hex_prefix,
    sniff_mime,
    verify_signature,
)
from src.utils.analysis_metrics import (
    analysis_errors_total,
    analysis_material_usage_total,
    analysis_parse_latency_budget_ratio,
    analysis_rejections_total,
    analysis_requests_total,
    analysis_stage_duration_seconds,
)

SUPPORTED_FILE_FORMATS = {
    "dxf",
    "dwg",
    "json",
    "step",
    "stp",
    "iges",
    "igs",
    "stl",
}


def _raise_unsupported_raw_dwg(file_name: str) -> None:
    analysis_requests_total.labels(status="error").inc()
    analysis_errors_total.labels(stage="input", code="unsupported_raw_dwg").inc()
    err = build_error(
        ErrorCode.UNSUPPORTED_INPUT_DWG,
        stage="input",
        message=(
            "Raw DWG input is not supported by /analyze. Convert through an "
            "approved DWG route before analysis."
        ),
        file=file_name,
        format="dwg",
    )
    raise HTTPException(status_code=415, detail=err)


def _is_empty_parser_stub_for_non_dxf_payload(doc: CadDocument, content: bytes) -> bool:
    head = content[:4096].upper()
    return (
        doc.metadata.get("parser") == "stub"
        and doc.entity_count() == 0
        and not doc.layers
        and b"SECTION" not in head
    )


async def run_document_pipeline(
    *,
    file_name: str,
    content: bytes,
    started_at: float,
    material: Optional[str] = None,
    project_id: Optional[str] = None,
    adapter_factory_cls: Optional[Any] = None,
) -> Dict[str, Any]:
    mime, reliable = sniff_mime(content[:4096])
    if reliable and not is_supported_mime(mime):
        analysis_rejections_total.labels(reason="mime_mismatch").inc()
        err = build_error(
            ErrorCode.INPUT_FORMAT_INVALID,
            stage="input",
            message=f"Unsupported MIME type: {mime}",
            mime=mime,
        )
        raise HTTPException(status_code=415, detail=err)

    max_mb = float(os.getenv("ANALYSIS_MAX_FILE_MB", "10"))
    size_mb = len(content) / (1024 * 1024)
    if size_mb > max_mb:
        analysis_requests_total.labels(status="error").inc()
        analysis_errors_total.labels(stage="input", code="file_too_large").inc()
        err = build_error(
            ErrorCode.INPUT_SIZE_EXCEEDED,
            stage="input",
            message="File too large",
            size_mb=round(size_mb, 3),
            max_mb=max_mb,
        )
        raise HTTPException(status_code=413, detail=err)

    if not content:
        err = build_error(
            ErrorCode.INPUT_ERROR,
            stage="input",
            message="Empty file",
        )
        raise HTTPException(status_code=400, detail=err)

    file_format = file_name.split(".")[-1].lower()
    if file_format not in SUPPORTED_FILE_FORMATS:
        analysis_requests_total.labels(status="error").inc()
        analysis_errors_total.labels(stage="input", code="unsupported_format").inc()
        err = build_error(
            ErrorCode.UNSUPPORTED_FORMAT,
            stage="input",
            message=f"Unsupported file format: {file_format}",
            format=file_format,
        )
        raise HTTPException(status_code=400, detail=err)

    if file_format == "dwg":
        _raise_unsupported_raw_dwg(file_name)

    if adapter_factory_cls is None:
        from src.adapters.factory import AdapterFactory as adapter_factory_cls

    adapter = adapter_factory_cls.get_adapter(file_format)
    parse_timeout = float(os.getenv("PARSE_TIMEOUT_SECONDS", "10"))
    parse_started_at = _time.time()
    try:
        doc: CadDocument
        if hasattr(adapter, "parse"):
            doc = await asyncio.wait_for(
                adapter.parse(content, file_name=file_name),  # type: ignore[attr-defined]
                timeout=parse_timeout,
            )
        else:
            await asyncio.wait_for(
                adapter.convert(content, file_name=file_name),
                timeout=parse_timeout,
            )
            doc = CadDocument(file_name=file_name, format=file_format)
            doc.metadata.update({"legacy": True})
        unified_data = doc.to_unified_dict()
    except asyncio.TimeoutError:
        from src.utils.analysis_metrics import parse_timeout_total

        parse_timeout_total.inc()
        analysis_errors_total.labels(stage="parse", code="timeout").inc()
        err = build_error(
            ErrorCode.TIMEOUT,
            stage="parse",
            message="Parse stage timeout",
            timeout_seconds=parse_timeout,
            file=file_name,
        )
        raise HTTPException(status_code=504, detail=err)
    except Exception:
        doc = CadDocument(file_name=file_name, format=file_format)
        unified_data = doc.to_unified_dict()

    if file_format == "dxf" and _is_empty_parser_stub_for_non_dxf_payload(doc, content):
        analysis_requests_total.labels(status="error").inc()
        analysis_errors_total.labels(stage="parse", code="empty_parser_stub").inc()
        err = build_error(
            ErrorCode.PARSE_FAILED,
            stage="parse",
            message="CAD parser returned an empty stub document",
            file=file_name,
            format=file_format,
        )
        raise HTTPException(status_code=422, detail=err)

    try:
        from src.utils.analysis_metrics import parse_stage_latency_seconds

        parse_stage_latency_seconds.labels(format=file_format).observe(
            _time.time() - parse_started_at
        )
    except Exception:
        pass

    valid_sig, expectation = verify_signature(content[:256], file_format)
    if not valid_sig:
        from src.utils.analysis_metrics import signature_validation_fail_total

        signature_validation_fail_total.labels(format=file_format).inc()
        analysis_rejections_total.labels(reason="signature_mismatch").inc()
        err = build_error(
            ErrorCode.INPUT_FORMAT_INVALID,
            stage="input",
            message="Signature validation failed",
            format=file_format,
            signature_prefix=signature_hex_prefix(content[:32]),
            expected_signature=expectation,
        )
        raise HTTPException(status_code=415, detail=err)

    strict_mode = os.getenv("FORMAT_STRICT_MODE", "0") == "1"
    from src.utils.analysis_metrics import (
        format_validation_fail_total,
        strict_mode_enabled,
    )

    if strict_mode:
        strict_mode_enabled.set(1)
        ok_deep, reason_deep = deep_format_validate(content[:2048], file_format)
        if not ok_deep:
            format_validation_fail_total.labels(
                format=file_format, reason=reason_deep
            ).inc()
            analysis_rejections_total.labels(reason="deep_format_invalid").inc()
            err = build_error(
                ErrorCode.INPUT_FORMAT_INVALID,
                stage="input",
                message="Deep format validation failed",
                format=file_format,
                reason=reason_deep,
            )
            raise HTTPException(status_code=415, detail=err)

        ok_matrix, reason_matrix = matrix_validate(content[:4096], file_format, project_id)
        if not ok_matrix:
            format_validation_fail_total.labels(
                format=file_format, reason=reason_matrix
            ).inc()
            analysis_rejections_total.labels(reason="matrix_format_invalid").inc()
            err = build_error(
                ErrorCode.INPUT_FORMAT_INVALID,
                stage="input",
                message="Matrix format validation failed",
                format=file_format,
                reason=reason_matrix,
                project_id=project_id,
            )
            raise HTTPException(status_code=415, detail=err)
    else:
        strict_mode_enabled.set(0)

    if material:
        doc.metadata["material"] = material
        analysis_material_usage_total.labels(material=material).inc()
    if project_id:
        doc.metadata["project_id"] = project_id

    parse_stage_duration = _time.time() - started_at
    analysis_stage_duration_seconds.labels(stage="parse").observe(parse_stage_duration)

    target_ms = float(os.getenv("ANALYSIS_PARSE_P95_TARGET_MS", "250"))
    if target_ms > 0:
        ratio = (parse_stage_duration * 1000.0) / target_ms
        analysis_parse_latency_budget_ratio.observe(ratio)

    max_entities = int(os.getenv("ANALYSIS_MAX_ENTITIES", "50000"))
    if doc.entity_count() > max_entities:
        analysis_rejections_total.labels(reason="entity_count_exceeded").inc()
        err = build_error(
            ErrorCode.VALIDATION_FAILED,
            stage="input",
            message="Entity count exceeds limit",
            entity_count=doc.entity_count(),
            max_entities=max_entities,
        )
        raise HTTPException(status_code=422, detail=err)

    return {
        "doc": doc,
        "file_format": file_format,
        "unified_data": unified_data,
        "parse_stage_duration": parse_stage_duration,
    }


__all__ = ["SUPPORTED_FILE_FORMATS", "run_document_pipeline"]
