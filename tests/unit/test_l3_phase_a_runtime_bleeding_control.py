"""L3 Phase A runtime bleeding-control: /model/reload sealed + default 'test' creds fail-closed.

Fail-closed, risk-reducing only — no request/credential/env opens an activation, and dev/CI keep the
historical 'test' default so the suite is not bricked. This is NOT the proof membrane (unbuilt); it
is the interim #509-style default: refuse.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _clear_posture(monkeypatch):
    for v in ("REQUIRE_STRONG_AUTH", "ENVIRONMENT", "APP_ENV", "ENV", "ADMIN_TOKEN", "API_KEY"):
        monkeypatch.delenv(v, raising=False)


# --- /model/reload is sealed (403), loader never reached -----------------------------------------
def test_model_reload_route_sealed_403_and_never_loads(monkeypatch) -> None:
    _clear_posture(monkeypatch)  # dev posture -> the 'test' creds pass the deps, so we reach the seal
    import src.ml.classifier as classifier

    calls = {"n": 0}
    monkeypatch.setattr(classifier, "reload_model",
                        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1) or {"status": "success"}))
    from fastapi.testclient import TestClient
    from src.main import app

    client = TestClient(app)
    resp = client.post(
        "/api/v1/model/reload",
        headers={"X-API-Key": "test", "X-Admin-Token": "test"},
        json={"path": "/tmp/evil.pkl", "force": True},
    )
    assert resp.status_code == 403
    assert "fail-closed" in resp.json()["detail"]
    assert calls["n"] == 0  # the caller-path loader is NEVER invoked through the route


def test_model_reload_seal_has_no_payload_bypass(monkeypatch) -> None:
    _clear_posture(monkeypatch)
    import src.ml.classifier as classifier

    monkeypatch.setattr(classifier, "reload_model",
                        lambda *a, **k: pytest.fail("reload_model reached through the sealed route"))
    from fastapi.testclient import TestClient
    from src.main import app

    client = TestClient(app)
    for payload in ({"path": None}, {"path": "", "force": True}, {"path": "/etc/shadow"}):
        r = client.post("/api/v1/model/reload",
                        headers={"X-API-Key": "test", "X-Admin-Token": "test"}, json=payload)
        assert r.status_code == 403, payload


# --- default 'test' credentials fail-closed in a production posture ------------------------------
def test_admin_token_refuses_unset_in_production(monkeypatch) -> None:
    from src.api.dependencies import get_admin_token
    _clear_posture(monkeypatch)
    monkeypatch.setenv("REQUIRE_STRONG_AUTH", "1")  # ADMIN_TOKEN unset
    with pytest.raises(HTTPException) as e:
        _run(get_admin_token("anything"))
    assert e.value.status_code == 500  # refuses to OPERATE with an unset admin token


def test_admin_token_refuses_default_test_in_production(monkeypatch) -> None:
    from src.api.dependencies import get_admin_token
    _clear_posture(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ADMIN_TOKEN", "test")  # the insecure default value
    with pytest.raises(HTTPException) as e:
        _run(get_admin_token("test"))
    assert e.value.status_code == 500


def test_admin_token_accepts_strong_secret_in_production(monkeypatch) -> None:
    from src.api.dependencies import get_admin_token
    _clear_posture(monkeypatch)
    monkeypatch.setenv("REQUIRE_STRONG_AUTH", "1")
    monkeypatch.setenv("ADMIN_TOKEN", "a-real-strong-secret")
    assert _run(get_admin_token("a-real-strong-secret")) == "a-real-strong-secret"
    with pytest.raises(HTTPException) as e:
        _run(get_admin_token("test"))
    assert e.value.status_code == 403  # wrong token still 403


def test_api_key_rejects_default_test_in_production(monkeypatch) -> None:
    from src.api.dependencies import get_api_key
    _clear_posture(monkeypatch)
    monkeypatch.setenv("REQUIRE_STRONG_AUTH", "1")
    with pytest.raises(HTTPException) as e:
        _run(get_api_key("test"))
    assert e.value.status_code == 401


def test_dev_posture_preserves_test_default(monkeypatch) -> None:
    # no prod flag / env -> historical dev default preserved, suite not bricked.
    from src.api.dependencies import get_api_key, get_admin_token
    _clear_posture(monkeypatch)
    assert _run(get_api_key("test")) == "test"
    assert _run(get_admin_token("test")) == "test"
