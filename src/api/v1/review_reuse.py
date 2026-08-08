"""HTTP API for ReviewReuse workbench — `/api/v1/review-reuse/*`."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key
from src.core.review_reuse.models import HumanDecisionState, TaskEventType
from src.core.review_reuse.service import (
    ReviewReuseError,
    ReviewReuseService,
    get_review_reuse_service,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _tenant_id(request: Request, api_key: str) -> str:
    # Prefer JWT/middleware tenant when present; else stable hash of API key.
    tid = getattr(request.state, "tenant_id", None)
    if tid:
        return str(tid)
    return "ak-" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def _reviewer_id(request: Request, api_key: str) -> tuple[str, bool]:
    """Return (reviewer_id, validated).

    ``validated=True`` only when middleware set JWT/integration subject
    (``user_id`` / ``auth_subject``). API-key fallback is never validated.
    """
    uid = getattr(request.state, "user_id", None) or getattr(request.state, "auth_subject", None)
    if uid:
        return str(uid), True
    # API-key-only path: not a trusted human identity for pilot; still stable for tests.
    return "ak-user-" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12], False


def _svc() -> ReviewReuseService:
    return get_review_reuse_service()


def _http(err: ReviewReuseError) -> HTTPException:
    status = 400
    if err.code == "not_found":
        status = 404
    elif err.code in ("decisions_disabled", "reviewer_not_validated"):
        status = 403
    elif err.code in ("already_decided",):
        status = 409
    return HTTPException(status_code=status, detail={"code": err.code, "message": err.message})


class TaskSummary(BaseModel):
    task_id: str
    tenant_id: str
    status: str
    created_at: float
    updated_at: float
    source_file_name: str
    trace_id: str
    candidate_count: int = 0


class DecisionRequest(BaseModel):
    state: HumanDecisionState
    reason_codes: List[str] = Field(default_factory=list)
    reason_text: str = ""
    candidate_id: Optional[str] = None
    idempotency_key: Optional[str] = None


@router.post("/tasks", response_model=Dict[str, Any])
async def create_task(
    request: Request,
    file: UploadFile = File(...),
    idempotency_key: Optional[str] = Form(default=None),
    api_key: str = Depends(get_api_key),
    service: ReviewReuseService = Depends(_svc),
) -> Dict[str, Any]:
    raw = await file.read()
    try:
        task = service.create_task(
            tenant_id=_tenant_id(request, api_key),
            file_name=file.filename or "upload.bin",
            file_bytes=raw,
            idempotency_key=idempotency_key,
        )
    except ReviewReuseError as exc:
        raise _http(exc) from exc
    return task.model_dump()


@router.get("/tasks", response_model=List[TaskSummary])
async def list_tasks(
    request: Request,
    api_key: str = Depends(get_api_key),
    service: ReviewReuseService = Depends(_svc),
) -> List[TaskSummary]:
    tasks = service.list_tasks(_tenant_id(request, api_key))
    return [
        TaskSummary(
            task_id=t.task_id,
            tenant_id=t.tenant_id,
            status=t.status.value,
            created_at=t.created_at,
            updated_at=t.updated_at,
            source_file_name=t.source_file_name,
            trace_id=t.trace_id,
            candidate_count=len(t.candidates),
        )
        for t in tasks
    ]


@router.get("/tasks/{task_id}", response_model=Dict[str, Any])
async def get_task(
    task_id: str,
    request: Request,
    api_key: str = Depends(get_api_key),
    service: ReviewReuseService = Depends(_svc),
) -> Dict[str, Any]:
    try:
        task = service.get_task(_tenant_id(request, api_key), task_id)
    except ReviewReuseError as exc:
        raise _http(exc) from exc
    return task.model_dump()


@router.post("/tasks/{task_id}/cancel", response_model=Dict[str, Any])
async def cancel_task(
    task_id: str,
    request: Request,
    api_key: str = Depends(get_api_key),
    service: ReviewReuseService = Depends(_svc),
) -> Dict[str, Any]:
    try:
        task = service.cancel(_tenant_id(request, api_key), task_id)
    except ReviewReuseError as exc:
        raise _http(exc) from exc
    return task.model_dump()


@router.get("/tasks/{task_id}/events", response_model=List[Dict[str, Any]])
async def list_events(
    task_id: str,
    request: Request,
    api_key: str = Depends(get_api_key),
    service: ReviewReuseService = Depends(_svc),
) -> List[Dict[str, Any]]:
    try:
        events = service.get_events(_tenant_id(request, api_key), task_id)
    except ReviewReuseError as exc:
        raise _http(exc) from exc
    return [e.model_dump() for e in events]


@router.get("/tasks/{task_id}/evidence-pack")
async def get_evidence_pack(
    task_id: str,
    request: Request,
    format: str = Query(default="json", pattern="^(json|markdown)$"),
    api_key: str = Depends(get_api_key),
    service: ReviewReuseService = Depends(_svc),
) -> Any:
    try:
        pack, md = service.get_evidence_pack(
            _tenant_id(request, api_key), task_id, as_markdown=(format == "markdown")
        )
    except ReviewReuseError as exc:
        raise _http(exc) from exc
    if format == "markdown":
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(md or "", media_type="text/markdown")
    return pack


@router.get("/metrics", response_model=Dict[str, Any])
async def review_metrics(
    request: Request,
    api_key: str = Depends(get_api_key),
    service: ReviewReuseService = Depends(_svc),
) -> Dict[str, Any]:
    """Review-workflow metrics (not Track E model-release metrics)."""
    return service.metrics(_tenant_id(request, api_key))


@router.get("/tasks/{task_id}/audit-export", response_model=Dict[str, Any])
async def audit_export(
    task_id: str,
    request: Request,
    api_key: str = Depends(get_api_key),
    service: ReviewReuseService = Depends(_svc),
) -> Dict[str, Any]:
    """Quarantined audit bundle (task + events + EvidencePack). Not training data."""
    try:
        return service.export_audit_bundle(_tenant_id(request, api_key), task_id)
    except ReviewReuseError as exc:
        raise _http(exc) from exc


@router.post("/tasks/{task_id}/decision", response_model=Dict[str, Any])
async def submit_decision(
    task_id: str,
    body: DecisionRequest,
    request: Request,
    api_key: str = Depends(get_api_key),
    service: ReviewReuseService = Depends(_svc),
) -> Dict[str, Any]:
    reviewer_id, validated = _reviewer_id(request, api_key)
    try:
        task = service.submit_decision(
            tenant_id=_tenant_id(request, api_key),
            task_id=task_id,
            state=body.state,
            reviewer_id=reviewer_id,
            reason_codes=body.reason_codes,
            reason_text=body.reason_text,
            candidate_id=body.candidate_id,
            idempotency_key=body.idempotency_key,
            reviewer_validated=validated,
        )
    except ReviewReuseError as exc:
        raise _http(exc) from exc
    return task.model_dump()
