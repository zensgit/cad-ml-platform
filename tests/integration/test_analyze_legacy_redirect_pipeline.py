from __future__ import annotations

from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.main import app


def test_analyze_legacy_redirect_get_route_delegates(monkeypatch):
    captured: dict[str, str] = {}

    def fake_raise_legacy_redirect(*, old_path: str, new_path: str, method: str = "GET"):
        captured["old_path"] = old_path
        captured["new_path"] = new_path
        captured["method"] = method
        raise HTTPException(status_code=410, detail={"code": "RESOURCE_GONE"})

    monkeypatch.setattr("src.api.v1.analyze.raise_legacy_redirect", fake_raise_legacy_redirect)

    client = TestClient(app)
    response = client.get("/api/v1/analyze/faiss/health", headers={"api-key": "test"})

    assert response.status_code == 410
    assert captured == {
        "old_path": "/api/v1/analyze/faiss/health",
        "new_path": "/api/v1/health/faiss",
        "method": "GET",
    }


def test_analyze_legacy_redirect_post_route_delegates(monkeypatch):
    captured: dict[str, str] = {}

    def fake_raise_legacy_redirect(*, old_path: str, new_path: str, method: str = "GET"):
        captured["old_path"] = old_path
        captured["new_path"] = new_path
        captured["method"] = method
        raise HTTPException(status_code=410, detail={"code": "RESOURCE_GONE"})

    monkeypatch.setattr("src.api.v1.analyze.raise_legacy_redirect", fake_raise_legacy_redirect)

    client = TestClient(app)
    response = client.post(
        "/api/v1/analyze/model/reload",
        json={"path": "/tmp/model.bin"},
        headers={"api-key": "test"},
    )

    assert response.status_code == 410
    assert captured == {
        "old_path": "/api/v1/analyze/model/reload",
        "new_path": "/api/v1/model/reload",
        "method": "POST",
    }


def test_analyze_legacy_redirect_delete_route_delegates(monkeypatch):
    captured: dict[str, str] = {}

    def fake_raise_legacy_redirect(*, old_path: str, new_path: str, method: str = "GET"):
        captured["old_path"] = old_path
        captured["new_path"] = new_path
        captured["method"] = method
        raise HTTPException(status_code=410, detail={"code": "RESOURCE_GONE"})

    monkeypatch.setattr("src.api.v1.analyze.raise_legacy_redirect", fake_raise_legacy_redirect)

    client = TestClient(app)
    response = client.delete(
        "/api/v1/analyze/vectors/orphans?threshold=0&dry_run=true",
        headers={"api-key": "test"},
    )

    assert response.status_code == 410
    assert captured == {
        "old_path": "/api/v1/analyze/vectors/orphans",
        "new_path": "/api/v1/maintenance/orphans",
        "method": "DELETE",
    }
