"""Verify admin token rotation invalidates old token and accepts new token."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.main import app


def test_admin_token_rotation(monkeypatch):
    client = TestClient(app)

    monkeypatch.setenv("ADMIN_TOKEN", "old-token")
    ok_resp = client.get(
        "/api/v1/model/opcode-audit",
        headers={"X-API-Key": "test", "X-Admin-Token": "old-token"},
    )
    assert ok_resp.status_code == 200

    # Rotate token
    monkeypatch.setenv("ADMIN_TOKEN", "new-token")

    old_resp = client.get(
        "/api/v1/model/opcode-audit",
        headers={"X-API-Key": "test", "X-Admin-Token": "old-token"},
    )
    assert old_resp.status_code == 403

    new_resp = client.get(
        "/api/v1/model/opcode-audit",
        headers={"X-API-Key": "test", "X-Admin-Token": "new-token"},
    )
    assert new_resp.status_code == 200
