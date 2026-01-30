"""Audit logging middleware for FastAPI."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.audit.service import (
    AuditAction,
    AuditActor,
    AuditLevel,
    AuditResource,
    get_audit_logger,
)


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware for automatic API audit logging."""

    def __init__(
        self,
        app: Any,
        exclude_paths: Optional[list[str]] = None,
        include_request_body: bool = False,
        include_response_body: bool = False,
        log_level_by_status: Optional[dict[int, AuditLevel]] = None,
    ):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/health",
            "/healthz",
            "/ready",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]
        self.include_request_body = include_request_body
        self.include_response_body = include_response_body
        self.log_level_by_status = log_level_by_status or {
            200: AuditLevel.INFO,
            201: AuditLevel.INFO,
            204: AuditLevel.INFO,
            400: AuditLevel.WARNING,
            401: AuditLevel.WARNING,
            403: AuditLevel.WARNING,
            404: AuditLevel.INFO,
            429: AuditLevel.WARNING,
            500: AuditLevel.ERROR,
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log audit event."""
        path = request.url.path

        # Skip excluded paths
        for exclude in self.exclude_paths:
            if path.startswith(exclude):
                return await call_next(request)

        # Record start time
        start_time = time.time()

        # Get request details
        request_id = request.headers.get("X-Request-ID", "")
        correlation_id = request.headers.get("X-Correlation-ID", "")

        # Create actor
        actor = self._create_actor(request)

        # Create resource
        resource = AuditResource(
            type="api_endpoint",
            id=path,
            name=f"{request.method} {path}",
            attributes={
                "method": request.method,
                "path": path,
                "query_params": dict(request.query_params),
            },
        )

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Determine outcome and level
        outcome = "success" if response.status_code < 400 else "failure"
        level = self._get_log_level(response.status_code)

        # Build details
        details = {
            "method": request.method,
            "path": path,
            "status_code": response.status_code,
            "query_params": dict(request.query_params),
        }

        # Log audit event
        audit_logger = get_audit_logger()
        await audit_logger.log(
            action=AuditAction.API_CALL,
            actor=actor,
            outcome=outcome,
            resource=resource,
            details=details,
            level=level,
            duration_ms=duration_ms,
            correlation_id=correlation_id or None,
            request_id=request_id or None,
        )

        return response

    def _create_actor(self, request: Request) -> AuditActor:
        """Create audit actor from request."""
        user_id = request.headers.get("X-User-ID", "anonymous")
        tenant_id = request.headers.get("X-Tenant-ID")
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "")

        return AuditActor(
            id=user_id,
            type="user" if user_id != "anonymous" else "anonymous",
            tenant_id=tenant_id,
            ip_address=client_ip,
            user_agent=user_agent[:200],
        )

    def _get_log_level(self, status_code: int) -> AuditLevel:
        """Get log level based on status code."""
        if status_code in self.log_level_by_status:
            return self.log_level_by_status[status_code]

        if status_code >= 500:
            return AuditLevel.ERROR
        elif status_code >= 400:
            return AuditLevel.WARNING
        else:
            return AuditLevel.INFO
