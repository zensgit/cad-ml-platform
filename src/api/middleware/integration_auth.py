from __future__ import annotations

from typing import Optional

from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


_PUBLIC_PREFIXES = (
    "/api/v1/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/metrics",
)


def _extract_bearer_token(auth_header: Optional[str]) -> Optional[str]:
    if not auth_header:
        return None
    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def _is_public_path(path: str) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in _PUBLIC_PREFIXES)


class IntegrationAuthMiddleware(BaseHTTPMiddleware):
    """Optional JWT validation for upstream platform integrations."""

    def __init__(self, app, *, settings) -> None:
        super().__init__(app)
        self.settings = settings
        mode_value = getattr(settings, "INTEGRATION_AUTH_MODE", "disabled") or "disabled"
        mode = mode_value.strip().lower()
        self.mode = mode if mode in {"disabled", "optional", "required"} else "disabled"
        self.jwt_secret = getattr(settings, "INTEGRATION_JWT_SECRET", "") or None
        self.jwt_alg = getattr(settings, "INTEGRATION_JWT_ALG", "HS256") or "HS256"
        self.tenant_header = getattr(settings, "INTEGRATION_TENANT_HEADER", "x-tenant-id")
        self.org_header = getattr(settings, "INTEGRATION_ORG_HEADER", "x-org-id")
        self.user_header = getattr(settings, "INTEGRATION_USER_HEADER", "x-user-id")

    async def dispatch(self, request: Request, call_next) -> Response:
        if self.mode == "disabled" or _is_public_path(request.url.path):
            self._set_state_from_headers(request)
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        token = _extract_bearer_token(auth_header)
        if not token:
            if self.mode == "required":
                return JSONResponse({"detail": "Missing bearer token"}, status_code=401)
            self._set_state_from_headers(request)
            return await call_next(request)

        if not self.jwt_secret:
            if self.mode == "required":
                return JSONResponse(
                    {"detail": "INTEGRATION_JWT_SECRET not configured"},
                    status_code=401,
                )
            self._set_state_from_headers(request)
            return await call_next(request)

        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_alg])
        except JWTError:
            return JSONResponse({"detail": "Invalid bearer token"}, status_code=401)

        tenant_claim = payload.get("tenant_id")
        subject = payload.get("sub")
        org_claim = payload.get("org_id")
        if not tenant_claim or not subject:
            return JSONResponse({"detail": "Invalid token claims"}, status_code=401)

        tenant_header = request.headers.get(self.tenant_header)
        if tenant_header and str(tenant_header) != str(tenant_claim):
            return JSONResponse({"detail": "Tenant mismatch"}, status_code=401)

        org_header = request.headers.get(self.org_header)
        if org_header and org_claim and str(org_header) != str(org_claim):
            return JSONResponse({"detail": "Org mismatch"}, status_code=401)

        tenant_id = tenant_header or str(tenant_claim)
        org_id = org_header or (str(org_claim) if org_claim else None)
        user_id = request.headers.get(self.user_header) or str(subject)

        request.state.tenant_id = tenant_id
        request.state.org_id = org_id
        request.state.user_id = user_id
        request.state.auth_subject = subject

        return await call_next(request)

    def _set_state_from_headers(self, request: Request) -> None:
        tenant_id = request.headers.get(self.tenant_header)
        org_id = request.headers.get(self.org_header)
        user_id = request.headers.get(self.user_header)
        if tenant_id:
            request.state.tenant_id = tenant_id
        if org_id:
            request.state.org_id = org_id
        if user_id:
            request.state.user_id = user_id
