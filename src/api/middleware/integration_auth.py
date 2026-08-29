"""Integration JWT middleware — production identity fail-closed (#517)."""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import jwt
    from jwt import PyJWTError
except Exception:  # pragma: no cover - optional dependency in test/runtime
    jwt = None  # type: ignore

    class PyJWTError(Exception):
        pass


from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_PUBLIC_PREFIXES = (
    "/api/v1/health",
    "/health",
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
    return any(
        path == prefix or path.startswith(f"{prefix}/") for prefix in _PUBLIC_PREFIXES
    )


class IntegrationAuthMiddleware(BaseHTTPMiddleware):
    """JWT validation for upstream platform integrations.

    Design-lock:
    - Identity (``request.state.user_id``) is **only** the validated token ``sub``.
    - ``x-user-id`` never establishes or overrides identity.
    - Header/claim mismatch → 401 in authenticated paths.
    - ``disabled`` / optional-without-token must not populate trusted identity
      from raw headers (hints only if ever needed — we set nothing trusted).
    """

    def __init__(self, app, *, settings) -> None:
        super().__init__(app)
        self.settings = settings
        mode_value = (
            getattr(settings, "INTEGRATION_AUTH_MODE", "disabled") or "disabled"
        )
        mode = mode_value.strip().lower()
        self.mode = mode if mode in {"disabled", "optional", "required"} else "disabled"
        self.jwt_secret = getattr(settings, "INTEGRATION_JWT_SECRET", "") or None
        self.jwt_alg = getattr(settings, "INTEGRATION_JWT_ALG", "HS256") or "HS256"
        self.jwt_audience = getattr(settings, "INTEGRATION_JWT_AUDIENCE", "") or None
        self.jwt_issuer = getattr(settings, "INTEGRATION_JWT_ISSUER", "") or None
        self.tenant_header = getattr(
            settings, "INTEGRATION_TENANT_HEADER", "x-tenant-id"
        )
        self.org_header = getattr(settings, "INTEGRATION_ORG_HEADER", "x-org-id")
        self.user_header = getattr(settings, "INTEGRATION_USER_HEADER", "x-user-id")

    async def dispatch(self, request: Request, call_next) -> Response:
        if _is_public_path(request.url.path):
            # Public paths: never establish trusted identity from headers.
            return await call_next(request)

        if self.mode == "disabled":
            # No trusted identity from raw headers (design-lock §1.E).
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        token = _extract_bearer_token(auth_header)
        if not token:
            if self.mode == "required":
                return JSONResponse({"detail": "Missing bearer token"}, status_code=401)
            # optional without token: do not set trusted identity from headers
            return await call_next(request)

        if not self.jwt_secret:
            if self.mode == "required":
                return JSONResponse(
                    {"detail": "INTEGRATION_JWT_SECRET not configured"},
                    status_code=401,
                )
            return await call_next(request)

        if jwt is None:
            if self.mode == "required":
                return JSONResponse({"detail": "PyJWT not installed"}, status_code=401)
            return await call_next(request)

        decode_kwargs: Dict[str, Any] = {
            "algorithms": [self.jwt_alg],
            "options": {"require": ["exp", "iat", "sub", "tenant_id"]},
        }
        if self.jwt_audience:
            decode_kwargs["audience"] = self.jwt_audience
        if self.jwt_issuer:
            decode_kwargs["issuer"] = self.jwt_issuer

        try:
            payload = jwt.decode(token, self.jwt_secret, **decode_kwargs)
        except PyJWTError:
            return JSONResponse({"detail": "Invalid bearer token"}, status_code=401)

        # If audience/issuer are configured they were verified above; if mode
        # is required and they were empty, boot refuse should have blocked.
        # Defense: required mode without aud/iss still needs exp/sub/tenant.
        if self.mode == "required":
            if not self.jwt_audience or not self.jwt_issuer:
                return JSONResponse(
                    {"detail": "JWT audience/issuer not configured"},
                    status_code=401,
                )

        tenant_claim = payload.get("tenant_id")
        subject = payload.get("sub")
        org_claim = payload.get("org_id")
        if (
            not isinstance(tenant_claim, str)
            or not tenant_claim
            or not isinstance(subject, str)
            or not subject
        ):
            return JSONResponse({"detail": "Invalid token claims"}, status_code=401)

        # Header mismatches are always rejected when a token is present.
        tenant_header = request.headers.get(self.tenant_header)
        if tenant_header and str(tenant_header) != str(tenant_claim):
            return JSONResponse({"detail": "Tenant mismatch"}, status_code=401)

        org_header = request.headers.get(self.org_header)
        if org_header and org_claim is not None and str(org_header) != str(org_claim):
            return JSONResponse({"detail": "Org mismatch"}, status_code=401)

        user_header = request.headers.get(self.user_header)
        if user_header and str(user_header) != str(subject):
            return JSONResponse({"detail": "User identity mismatch"}, status_code=401)

        # Identity ONLY from validated token claims (design-lock §1.D).
        request.state.tenant_id = tenant_claim
        request.state.org_id = str(org_claim) if org_claim is not None else None
        request.state.user_id = subject
        request.state.auth_subject = subject
        request.state.identity_provider = self.jwt_issuer
        request.state.review_reuse_identity_validated = bool(self.jwt_issuer)

        return await call_next(request)
