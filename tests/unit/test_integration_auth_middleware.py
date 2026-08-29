"""Integration auth middleware — production identity fail-closed goldens."""

from __future__ import annotations

import time
from types import SimpleNamespace

import jwt
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.middleware.integration_auth import IntegrationAuthMiddleware


def _make_settings(
    *,
    mode: str = "required",
    secret: str = "test-secret",
    alg: str = "HS256",
    audience: str = "cad-ml-api",
    issuer: str = "cad-ml-issuer",
) -> SimpleNamespace:
    return SimpleNamespace(
        INTEGRATION_AUTH_MODE=mode,
        INTEGRATION_JWT_SECRET=secret,
        INTEGRATION_JWT_ALG=alg,
        INTEGRATION_JWT_AUDIENCE=audience,
        INTEGRATION_JWT_ISSUER=issuer,
        INTEGRATION_TENANT_HEADER="x-tenant-id",
        INTEGRATION_ORG_HEADER="x-org-id",
        INTEGRATION_USER_HEADER="x-user-id",
    )


def _build_app(settings: SimpleNamespace) -> FastAPI:
    app = FastAPI()
    app.add_middleware(IntegrationAuthMiddleware, settings=settings)

    @app.get("/private")
    async def private(request: Request) -> dict[str, str | bool | None]:
        return {
            "tenant_id": getattr(request.state, "tenant_id", None),
            "org_id": getattr(request.state, "org_id", None),
            "user_id": getattr(request.state, "user_id", None),
            "auth_subject": getattr(request.state, "auth_subject", None),
            "identity_provider": getattr(request.state, "identity_provider", None),
            "review_reuse_identity_validated": getattr(
                request.state, "review_reuse_identity_validated", None
            ),
            "review_reuse_tenant_validated": getattr(
                request.state, "review_reuse_tenant_validated", None
            ),
        }

    return app


def _encode(payload: dict, secret: str, alg: str) -> str:
    token = jwt.encode(payload, secret, algorithm=alg)
    return token if isinstance(token, str) else token.decode("utf-8")


def _full_claims(**extra) -> dict:
    now = int(time.time())
    base = {
        "sub": "user-1",
        "tenant_id": "tenant-1",
        "iat": now,
        "exp": now + 3600,
        "aud": "cad-ml-api",
        "iss": "cad-ml-issuer",
    }
    base.update(extra)
    return base


def test_required_missing_token_rejected() -> None:
    client = TestClient(_build_app(_make_settings()), headers={"X-API-Key": "test"})
    assert client.get("/private").status_code == 401


def test_required_invalid_token_rejected() -> None:
    client = TestClient(_build_app(_make_settings()), headers={"X-API-Key": "test"})
    response = client.get("/private", headers={"Authorization": "Bearer not-a-jwt"})
    assert response.status_code == 401


def test_required_missing_claims_rejected() -> None:
    settings = _make_settings()
    token = _encode(
        {"sub": "user-1"}, settings.INTEGRATION_JWT_SECRET, settings.INTEGRATION_JWT_ALG
    )
    client = TestClient(_build_app(settings), headers={"X-API-Key": "test"})
    response = client.get("/private", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


@pytest.mark.parametrize(
    "claims",
    [
        _full_claims(tenant_id=123),
        _full_claims(sub=123),
    ],
)
def test_required_non_string_identity_claims_rejected(claims) -> None:
    settings = _make_settings()
    token = _encode(
        claims, settings.INTEGRATION_JWT_SECRET, settings.INTEGRATION_JWT_ALG
    )
    client = TestClient(_build_app(settings), headers={"X-API-Key": "test"})
    response = client.get("/private", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_required_tenant_mismatch_rejected() -> None:
    settings = _make_settings()
    token = _encode(
        _full_claims(), settings.INTEGRATION_JWT_SECRET, settings.INTEGRATION_JWT_ALG
    )
    client = TestClient(_build_app(settings), headers={"X-API-Key": "test"})
    response = client.get(
        "/private",
        headers={
            "Authorization": f"Bearer {token}",
            "x-tenant-id": "other-tenant",
        },
    )
    assert response.status_code == 401


def test_required_valid_token_sets_state_from_sub() -> None:
    """Identity is token sub — not a spoofable x-user-id header (design-lock D)."""
    settings = _make_settings()
    token = _encode(
        _full_claims(sub="user-1"),
        settings.INTEGRATION_JWT_SECRET,
        settings.INTEGRATION_JWT_ALG,
    )
    client = TestClient(_build_app(settings), headers={"X-API-Key": "test"})
    response = client.get(
        "/private",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "user-1"
    assert payload["auth_subject"] == "user-1"
    assert payload["tenant_id"] == "tenant-1"
    assert payload["identity_provider"] == "cad-ml-issuer"
    assert payload["review_reuse_identity_validated"] is True
    assert payload["review_reuse_tenant_validated"] is True


def test_optional_verified_token_without_issuer_still_validates_tenant() -> None:
    settings = _make_settings(mode="optional", issuer="")
    token = _encode(
        _full_claims(tenant_id="tenant-signed"),
        settings.INTEGRATION_JWT_SECRET,
        settings.INTEGRATION_JWT_ALG,
    )
    client = TestClient(_build_app(settings), headers={"X-API-Key": "shared-key"})

    response = client.get("/private", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["tenant_id"] == "tenant-signed"
    assert payload["review_reuse_tenant_validated"] is True
    assert payload["review_reuse_identity_validated"] is False
    assert payload["identity_provider"] is None


def test_required_user_header_mismatch_rejected() -> None:
    settings = _make_settings()
    token = _encode(
        _full_claims(sub="alice"),
        settings.INTEGRATION_JWT_SECRET,
        settings.INTEGRATION_JWT_ALG,
    )
    client = TestClient(_build_app(settings), headers={"X-API-Key": "test"})
    response = client.get(
        "/private",
        headers={
            "Authorization": f"Bearer {token}",
            "x-user-id": "bob",
        },
    )
    assert response.status_code == 401
    assert "mismatch" in response.json()["detail"].lower()


def test_jwt_without_exp_rejected() -> None:
    settings = _make_settings()
    claims = _full_claims()
    del claims["exp"]
    token = _encode(
        claims, settings.INTEGRATION_JWT_SECRET, settings.INTEGRATION_JWT_ALG
    )
    client = TestClient(_build_app(settings), headers={"X-API-Key": "test"})
    response = client.get("/private", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_jwt_wrong_audience_rejected() -> None:
    settings = _make_settings()
    token = _encode(
        _full_claims(aud="other-aud"),
        settings.INTEGRATION_JWT_SECRET,
        settings.INTEGRATION_JWT_ALG,
    )
    client = TestClient(_build_app(settings), headers={"X-API-Key": "test"})
    response = client.get("/private", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_jwt_wrong_issuer_rejected() -> None:
    settings = _make_settings()
    token = _encode(
        _full_claims(iss="other-iss"),
        settings.INTEGRATION_JWT_SECRET,
        settings.INTEGRATION_JWT_ALG,
    )
    client = TestClient(_build_app(settings), headers={"X-API-Key": "test"})
    response = client.get("/private", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_disabled_mode_does_not_set_trusted_identity_from_headers() -> None:
    settings = _make_settings(mode="disabled")
    client = TestClient(_build_app(settings), headers={"X-API-Key": "test"})
    response = client.get(
        "/private",
        headers={"x-user-id": "spoofed", "x-tenant-id": "t1"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] is None
    assert payload["tenant_id"] is None
    assert payload["review_reuse_identity_validated"] is None
    assert payload["review_reuse_tenant_validated"] is None


def test_optional_without_token_does_not_trust_headers() -> None:
    settings = _make_settings(mode="optional")
    client = TestClient(_build_app(settings), headers={"X-API-Key": "test"})
    response = client.get(
        "/private",
        headers={"x-user-id": "spoofed"},
    )
    assert response.status_code == 200
    assert response.json()["user_id"] is None
