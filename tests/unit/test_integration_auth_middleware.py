from __future__ import annotations

from types import SimpleNamespace

import jwt
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.middleware.integration_auth import IntegrationAuthMiddleware


def _make_settings(
    *,
    mode: str = "required",
    secret: str = "test-secret",
    alg: str = "HS256",
) -> SimpleNamespace:
    return SimpleNamespace(
        INTEGRATION_AUTH_MODE=mode,
        INTEGRATION_JWT_SECRET=secret,
        INTEGRATION_JWT_ALG=alg,
        INTEGRATION_TENANT_HEADER="x-tenant-id",
        INTEGRATION_ORG_HEADER="x-org-id",
        INTEGRATION_USER_HEADER="x-user-id",
    )


def _build_app(settings: SimpleNamespace) -> FastAPI:
    app = FastAPI()
    app.add_middleware(IntegrationAuthMiddleware, settings=settings)

    @app.get("/private")
    async def private(request: Request) -> dict[str, str | None]:
        return {
            "tenant_id": getattr(request.state, "tenant_id", None),
            "org_id": getattr(request.state, "org_id", None),
            "user_id": getattr(request.state, "user_id", None),
            "auth_subject": getattr(request.state, "auth_subject", None),
        }

    return app


def _encode(payload: dict[str, str], secret: str, alg: str) -> str:
    token = jwt.encode(payload, secret, algorithm=alg)
    return token if isinstance(token, str) else token.decode("utf-8")


def test_required_missing_token_rejected() -> None:
    settings = _make_settings()
    client = TestClient(_build_app(settings))
    response = client.get("/private")
    assert response.status_code == 401


def test_required_invalid_token_rejected() -> None:
    settings = _make_settings()
    client = TestClient(_build_app(settings))
    response = client.get("/private", headers={"Authorization": "Bearer not-a-jwt"})
    assert response.status_code == 401


def test_required_missing_claims_rejected() -> None:
    settings = _make_settings()
    token = _encode(
        {"sub": "user-1"},
        settings.INTEGRATION_JWT_SECRET,
        settings.INTEGRATION_JWT_ALG,
    )
    client = TestClient(_build_app(settings))
    response = client.get("/private", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_required_tenant_mismatch_rejected() -> None:
    settings = _make_settings()
    token = _encode(
        {"tenant_id": "tenant-1", "sub": "user-1"},
        settings.INTEGRATION_JWT_SECRET,
        settings.INTEGRATION_JWT_ALG,
    )
    client = TestClient(_build_app(settings))
    response = client.get(
        "/private",
        headers={"Authorization": f"Bearer {token}", "x-tenant-id": "tenant-2"},
    )
    assert response.status_code == 401


def test_required_valid_token_sets_state() -> None:
    settings = _make_settings()
    token = _encode(
        {"tenant_id": "tenant-1", "sub": "user-1", "org_id": "org-1"},
        settings.INTEGRATION_JWT_SECRET,
        settings.INTEGRATION_JWT_ALG,
    )
    client = TestClient(_build_app(settings))
    response = client.get(
        "/private",
        headers={
            "Authorization": f"Bearer {token}",
            "x-tenant-id": "tenant-1",
            "x-org-id": "org-1",
            "x-user-id": "user-header",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["tenant_id"] == "tenant-1"
    assert payload["org_id"] == "org-1"
    assert payload["user_id"] == "user-header"
    assert payload["auth_subject"] == "user-1"
