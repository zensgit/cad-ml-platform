"""HTTP API for ReviewReuse workbench — `/api/v1/review-reuse/*`."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies import get_api_key
from src.core.review_reuse.canonical import CanonicalJSONError, strict_json_loads
from src.core.review_reuse.models import HumanDecisionState
from src.core.review_reuse.service import (
    ReviewReuseError,
    ReviewReuseService,
    canonical_reviewer_principal,
    get_review_reuse_service,
    max_upload_bytes,
)

logger = logging.getLogger(__name__)


class ReviewReuseRoute(APIRoute):
    """Keep framework request errors inside the ReviewReuse error contract."""

    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        handler = super().get_route_handler()

        async def route_handler(request: Request) -> Response:
            try:
                content_type = request.headers.get("content-type", "").partition(";")[0]
                if content_type.strip().lower() == "application/json":
                    strict_json_loads(await request.body())
                return await handler(request)
            except CanonicalJSONError:
                return JSONResponse(
                    status_code=422,
                    content={
                        "detail": {
                            "code": "invalid_request",
                            "message": "ReviewReuse request validation failed",
                        }
                    },
                )
            except RequestValidationError:
                return JSONResponse(
                    status_code=422,
                    content={
                        "detail": {
                            "code": "invalid_request",
                            "message": "ReviewReuse request validation failed",
                        }
                    },
                )
            except ReviewReuseError as exc:
                return _http(exc)

        return route_handler


class ReviewReuseErrorDetail(BaseModel):
    code: str
    message: str


class ReviewReuseErrorResponse(BaseModel):
    detail: ReviewReuseErrorDetail


class PlatformAuthErrorResponse(BaseModel):
    detail: str


_OPENAPI_ERROR_RESPONSES = {
    400: {"model": ReviewReuseErrorResponse, "description": "Invalid tenant identity"},
    401: {
        "model": PlatformAuthErrorResponse,
        "description": "Existing platform authentication boundary",
    },
    403: {"model": ReviewReuseErrorResponse, "description": "Owner or identity gate"},
    404: {"model": ReviewReuseErrorResponse, "description": "Task not visible"},
    409: {"model": ReviewReuseErrorResponse, "description": "Ledger conflict"},
    413: {"model": ReviewReuseErrorResponse, "description": "Upload too large"},
    415: {"model": ReviewReuseErrorResponse, "description": "Unsupported file type"},
    422: {
        "model": ReviewReuseErrorResponse,
        "description": "Request validation failure",
    },
    500: {"model": ReviewReuseErrorResponse, "description": "Sanitized internal error"},
    503: {"model": ReviewReuseErrorResponse, "description": "Ledger unavailable"},
}


router = APIRouter(
    route_class=ReviewReuseRoute,
    responses=_OPENAPI_ERROR_RESPONSES,
)


def _tenant_identity(request: Request, api_key: str) -> tuple[str, bool]:
    validated = bool(getattr(request.state, "review_reuse_identity_validated", False))
    tenant_id = getattr(request.state, "tenant_id", None)
    if validated and tenant_id is not None:
        if not isinstance(tenant_id, str):
            raise ReviewReuseError(
                "tenant_invalid", "validated tenant claim is invalid"
            )
        literal = tenant_id
        if not literal or literal.strip() != literal or literal.startswith("ak-"):
            raise ReviewReuseError(
                "tenant_invalid", "validated tenant claim is invalid"
            )
        return literal, True
    fallback = "ak-" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return fallback, False


def _reviewer_identity(request: Request, api_key: str) -> tuple[str, str, bool]:
    validated = bool(getattr(request.state, "review_reuse_identity_validated", False))
    subject = getattr(request.state, "auth_subject", None)
    identity_provider = getattr(request.state, "identity_provider", None)
    if validated and isinstance(subject, str) and isinstance(identity_provider, str):
        principal = canonical_reviewer_principal(identity_provider, subject)
        return principal, "validated_principal", True
    fallback = "ak-user-" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
    return fallback, "api_key_fallback", False


def _svc() -> ReviewReuseService:
    return get_review_reuse_service()


_ERROR_STATUS = {
    "tenant_invalid": 400,
    "empty_input": 422,
    "invalid_decision": 422,
    "invalid_request": 422,
    "input_too_large": 413,
    "unsupported_file_type": 415,
    "not_found": 404,
    "decisions_disabled": 403,
    "tenant_not_validated": 403,
    "reviewer_not_validated": 403,
    "not_ready": 409,
    "invalid_state_transition": 409,
    "already_decided": 409,
    "idempotency_key_conflict": 409,
    "revision_conflict": 409,
    "store_index_corrupt": 503,
    "store_record_corrupt": 503,
    "store_writer_conflict": 503,
    "internal_error": 500,
}


def _http(err: ReviewReuseError) -> JSONResponse:
    status = _ERROR_STATUS.get(err.code)
    if status is None:
        logger.error("review_reuse_unmapped_error code=%s", err.code)
        status = 500
        code = "internal_error"
        message = "ReviewReuse request failed"
    else:
        code = err.code
        message = err.message
    return JSONResponse(
        status_code=status,
        content={"detail": {"code": code, "message": message}},
    )


async def _read_limited_upload(upload: UploadFile) -> bytes:
    limit = max_upload_bytes()
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(min(1024 * 1024, limit - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise ReviewReuseError(
                "input_too_large", "DXF input exceeds the upload limit"
            )
        chunks.append(chunk)
    return b"".join(chunks)


class TaskSummary(BaseModel):
    task_id: str
    tenant_id: str
    status: str
    revision: int
    created_at: float
    updated_at: float
    source_file_name: str
    trace_id: str
    candidate_count: int = 0


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: HumanDecisionState
    expected_revision: int = Field(ge=1, strict=True)
    evidence_pack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason_codes: List[str] = Field(default_factory=list)
    reason_text: str = ""
    candidate_id: Optional[str] = None
    idempotency_key: Optional[str] = Field(default=None, max_length=128)


@router.post("/tasks", response_model=Dict[str, Any])
async def create_task(
    request: Request,
    file: UploadFile = File(...),
    idempotency_key: Optional[str] = Form(default=None),
    api_key: str = Depends(get_api_key),
    service: ReviewReuseService = Depends(_svc),
) -> Any:
    try:
        tenant_id, _validated = _tenant_identity(request, api_key)
        raw = await _read_limited_upload(file)
        task = service.create_task(
            tenant_id=tenant_id,
            file_name=file.filename or "upload.bin",
            file_bytes=raw,
            idempotency_key=idempotency_key,
        )
    except ReviewReuseError as exc:
        return _http(exc)
    return task.model_dump()


@router.get("/tasks", response_model=List[TaskSummary])
async def list_tasks(
    request: Request,
    api_key: str = Depends(get_api_key),
    service: ReviewReuseService = Depends(_svc),
) -> Any:
    try:
        tenant_id, _validated = _tenant_identity(request, api_key)
        tasks = service.list_tasks(tenant_id)
    except ReviewReuseError as exc:
        return _http(exc)
    return [
        TaskSummary(
            task_id=task.task_id,
            tenant_id=task.tenant_id,
            status=task.status.value,
            revision=task.revision,
            created_at=task.created_at,
            updated_at=task.updated_at,
            source_file_name=task.source_file_name,
            trace_id=task.trace_id,
            candidate_count=len(task.candidates),
        )
        for task in tasks
    ]


@router.get("/tasks/{task_id}", response_model=Dict[str, Any])
async def get_task(
    task_id: str,
    request: Request,
    api_key: str = Depends(get_api_key),
    service: ReviewReuseService = Depends(_svc),
) -> Any:
    try:
        tenant_id, _validated = _tenant_identity(request, api_key)
        task = service.get_task(tenant_id, task_id)
    except ReviewReuseError as exc:
        return _http(exc)
    return task.model_dump()


@router.post("/tasks/{task_id}/cancel", response_model=Dict[str, Any])
async def cancel_task(
    task_id: str,
    request: Request,
    api_key: str = Depends(get_api_key),
    service: ReviewReuseService = Depends(_svc),
) -> Any:
    try:
        tenant_id, _validated = _tenant_identity(request, api_key)
        task = service.cancel(tenant_id, task_id)
    except ReviewReuseError as exc:
        return _http(exc)
    return task.model_dump()


@router.get("/tasks/{task_id}/events", response_model=List[Dict[str, Any]])
async def list_events(
    task_id: str,
    request: Request,
    api_key: str = Depends(get_api_key),
    service: ReviewReuseService = Depends(_svc),
) -> Any:
    try:
        tenant_id, _validated = _tenant_identity(request, api_key)
        events = service.get_events(tenant_id, task_id)
    except ReviewReuseError as exc:
        return _http(exc)
    return [event.model_dump() for event in events]


@router.get("/tasks/{task_id}/evidence-pack")
async def get_evidence_pack(
    task_id: str,
    request: Request,
    format: str = Query(default="json", pattern="^(json|markdown)$"),
    api_key: str = Depends(get_api_key),
    service: ReviewReuseService = Depends(_svc),
) -> Any:
    try:
        tenant_id, _validated = _tenant_identity(request, api_key)
        pack, markdown = service.get_evidence_pack(
            tenant_id, task_id, as_markdown=(format == "markdown")
        )
    except ReviewReuseError as exc:
        return _http(exc)
    if format == "markdown":
        return PlainTextResponse(markdown or "", media_type="text/markdown")
    return pack


@router.get("/metrics")
async def review_metrics(
    request: Request,
    format: str = Query(default="json", pattern="^(json|markdown)$"),
    api_key: str = Depends(get_api_key),
    service: ReviewReuseService = Depends(_svc),
) -> Any:
    """Review-workflow metrics, not Track E model-release metrics."""

    try:
        tenant_id, _validated = _tenant_identity(request, api_key)
        metrics = service.metrics(tenant_id)
    except ReviewReuseError as exc:
        return _http(exc)
    if format == "markdown":
        from src.core.review_reuse.metrics import format_metrics_markdown

        return PlainTextResponse(
            format_metrics_markdown(metrics), media_type="text/markdown"
        )
    return metrics


@router.get("/tasks/{task_id}/audit-export", response_model=Dict[str, Any])
async def audit_export(
    task_id: str,
    request: Request,
    api_key: str = Depends(get_api_key),
    service: ReviewReuseService = Depends(_svc),
) -> Any:
    """Quarantined audit bundle; never training data."""

    try:
        tenant_id, _validated = _tenant_identity(request, api_key)
        return service.export_audit_bundle(tenant_id, task_id)
    except ReviewReuseError as exc:
        return _http(exc)


@router.post("/tasks/{task_id}/decision", response_model=Dict[str, Any])
async def submit_decision(
    task_id: str,
    body: DecisionRequest,
    request: Request,
    api_key: str = Depends(get_api_key),
    service: ReviewReuseService = Depends(_svc),
) -> Any:
    try:
        tenant_id, tenant_validated = _tenant_identity(request, api_key)
        reviewer_id, reviewer_kind, reviewer_validated = _reviewer_identity(
            request, api_key
        )
        task = service.submit_decision(
            tenant_id=tenant_id,
            task_id=task_id,
            state=body.state,
            reviewer_id=reviewer_id,
            reviewer_kind=reviewer_kind,
            tenant_validated=tenant_validated,
            reviewer_validated=reviewer_validated,
            expected_revision=body.expected_revision,
            evidence_pack_sha256=body.evidence_pack_sha256,
            reason_codes=body.reason_codes,
            reason_text=body.reason_text,
            candidate_id=body.candidate_id,
            idempotency_key=body.idempotency_key,
        )
    except ReviewReuseError as exc:
        return _http(exc)
    return task.model_dump()
