"""L3 #517 production identity fail-closed — fail-first goldens."""

from __future__ import annotations

import asyncio
import os

import pytest
from fastapi import HTTPException


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clear_posture(monkeypatch):
    for v in (
        "REQUIRE_STRONG_AUTH",
        "ENVIRONMENT",
        "APP_ENV",
        "ENV",
        "ADMIN_TOKEN",
        "API_KEY",
        "API_KEYS",
        "X_API_KEY",
        "REVIEW_REUSE_DECISIONS_ENABLED",
    ):
        monkeypatch.delenv(v, raising=False)
    yield


def test_unset_env_is_production_posture(monkeypatch) -> None:
    from src.api.production_identity import is_development_opt_in, is_production_posture

    assert is_development_opt_in() is False
    assert is_production_posture() is True


def test_development_opt_in(monkeypatch) -> None:
    from src.api.production_identity import is_development_opt_in, is_production_posture

    monkeypatch.setenv("ENVIRONMENT", "development")
    assert is_development_opt_in() is True
    assert is_production_posture() is False


def test_api_key_missing_401(monkeypatch) -> None:
    from src.api.dependencies import get_api_key

    monkeypatch.setenv("ENVIRONMENT", "development")
    with pytest.raises(HTTPException) as e:
        _run(get_api_key(None))  # type: ignore[arg-type]
    assert e.value.status_code == 401


def test_api_key_missing_header_401_via_http_path(monkeypatch) -> None:
    """Golden: zero X-API-Key through real FastAPI Header() resolution (TestClient).

    Must NOT call get_api_key() as a bare coroutine — that bypasses Header(default=...)
    and cannot catch a regression reintroducing Header(default=\"test\").
    """
    from fastapi.testclient import TestClient

    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("API_KEY", "test")
    from src.main import app

    # Client with NO default auth headers — forces the real missing-header path.
    client = TestClient(app)
    # Pick a non-public route that requires get_api_key.
    resp = client.get(
        "/api/v1/tolerance/it", params={"diameter_mm": 25, "grade": "IT7"}
    )
    assert resp.status_code == 401
    assert "API Key" in resp.json().get("detail", "")


def test_api_key_wrong_header_401_via_http_path(monkeypatch) -> None:
    """Golden: attacker-chosen key through real HTTP path → 401."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("API_KEY", "test")
    from src.main import app

    client = TestClient(app)
    resp = client.get(
        "/api/v1/tolerance/it",
        params={"diameter_mm": 25, "grade": "IT7"},
        headers={"X-API-Key": "attacker-key"},
    )
    assert resp.status_code == 401


def test_api_key_attacker_chosen_401_in_production(monkeypatch) -> None:
    from src.api.dependencies import get_api_key

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("API_KEY", "real-secret")
    with pytest.raises(HTTPException) as e:
        _run(get_api_key("attacker-key"))
    assert e.value.status_code == 401


def test_api_key_rejects_default_test_in_production(monkeypatch) -> None:
    from src.api.dependencies import get_api_key

    monkeypatch.setenv("REQUIRE_STRONG_AUTH", "1")
    monkeypatch.setenv("API_KEY", "real-secret")
    with pytest.raises(HTTPException) as e:
        _run(get_api_key("test"))
    assert e.value.status_code == 401


def test_api_key_accepts_configured_secret(monkeypatch) -> None:
    from src.api.dependencies import get_api_key

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("API_KEY", "real-secret")
    assert _run(get_api_key("real-secret")) == "real-secret"


def test_dev_posture_accepts_harness_test_key(monkeypatch) -> None:
    from src.api.dependencies import get_admin_token, get_api_key

    monkeypatch.setenv("ENVIRONMENT", "development")
    # No API_KEY → harness default set {test}
    assert _run(get_api_key("test")) == "test"
    assert _run(get_admin_token("test")) == "test"


def test_admin_token_refuses_unset_in_production(monkeypatch) -> None:
    from src.api.dependencies import get_admin_token

    monkeypatch.setenv("REQUIRE_STRONG_AUTH", "1")
    with pytest.raises(HTTPException) as e:
        _run(get_admin_token("test"))
    assert e.value.status_code == 500


def test_admin_token_refuses_default_test_in_production(monkeypatch) -> None:
    from src.api.dependencies import get_admin_token

    monkeypatch.setenv("REQUIRE_STRONG_AUTH", "1")
    monkeypatch.setenv("ADMIN_TOKEN", "test")
    with pytest.raises(HTTPException) as e:
        _run(get_admin_token("test"))
    assert e.value.status_code == 500


def test_admin_token_accepts_strong_secret_in_production(monkeypatch) -> None:
    from src.api.dependencies import get_admin_token

    monkeypatch.setenv("REQUIRE_STRONG_AUTH", "1")
    monkeypatch.setenv("ADMIN_TOKEN", "a-real-strong-secret")
    assert _run(get_admin_token("a-real-strong-secret")) == "a-real-strong-secret"
    with pytest.raises(HTTPException) as e:
        _run(get_admin_token("test"))
    assert e.value.status_code == 403


def test_boot_refuses_disabled_integration_in_production(monkeypatch) -> None:
    from src.api.production_identity import validate_boot_identity

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("API_KEY", "real-api")
    monkeypatch.setenv("ADMIN_TOKEN", "real-admin")
    err = validate_boot_identity(integration_auth_mode="disabled")
    assert err is not None
    assert "disabled" in err


def test_boot_refuses_unset_api_key_in_production(monkeypatch) -> None:
    from src.api.production_identity import validate_boot_identity

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ADMIN_TOKEN", "real-admin")
    err = validate_boot_identity(
        integration_auth_mode="required",
        integration_jwt_secret="s",
        integration_jwt_audience="a",
        integration_jwt_issuer="i",
    )
    assert err is not None
    assert "API_KEY" in err


def test_boot_ok_with_full_production_config(monkeypatch) -> None:
    from src.api.production_identity import validate_boot_identity

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("API_KEY", "real-api")
    monkeypatch.setenv("ADMIN_TOKEN", "real-admin")
    err = validate_boot_identity(
        integration_auth_mode="required",
        integration_jwt_secret="s",
        integration_jwt_audience="aud",
        integration_jwt_issuer="iss",
    )
    assert err is None


def test_boot_ok_in_development_with_disabled_auth(monkeypatch) -> None:
    from src.api.production_identity import validate_boot_identity

    monkeypatch.setenv("ENVIRONMENT", "development")
    err = validate_boot_identity(integration_auth_mode="disabled")
    assert err is None


def test_decision_enabled_boot_requires_complete_validated_identity(
    monkeypatch,
) -> None:
    from src.api.production_identity import validate_boot_identity

    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "true")
    err = validate_boot_identity(integration_auth_mode="optional")
    assert err is not None
    assert "ReviewReuse decisions require" in err

    assert (
        validate_boot_identity(
            integration_auth_mode="required",
            integration_jwt_secret="secret",
            integration_jwt_audience="audience",
            integration_jwt_issuer="issuer",
        )
        is None
    )


def test_decision_enabled_boot_rejects_padded_issuer(monkeypatch) -> None:
    from src.api.production_identity import validate_boot_identity

    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "true")
    err = validate_boot_identity(
        integration_auth_mode="required",
        integration_jwt_secret="secret",
        integration_jwt_audience="audience",
        integration_jwt_issuer=" issuer ",
    )
    assert err is not None
    assert "issuer" in err.lower()


def test_pytest_without_opt_in_is_production(monkeypatch) -> None:
    """Golden: without ENVIRONMENT=development the posture is production."""
    from src.api.production_identity import is_production_posture

    for k in ("ENVIRONMENT", "APP_ENV", "ENV", "REQUIRE_STRONG_AUTH"):
        monkeypatch.delenv(k, raising=False)
    assert is_production_posture() is True
