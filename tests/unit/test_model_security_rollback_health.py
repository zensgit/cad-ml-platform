"""Tests: model security failure leads to rollback & health reflects failure state."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class DummyModel:
    def predict(self, vectors):  # type: ignore[no-untyped-def]
        return ["ok"] * len(vectors)


def _snapshot_classifier_state() -> Dict[str, Any]:
    import src.ml.classifier as classifier

    keys = [
        "_MODEL",
        "_MODEL_HASH",
        "_MODEL_VERSION",
        "_MODEL_PATH",
        "_MODEL_LOADED_AT",
        "_MODEL_LAST_ERROR",
        "_MODEL_LOAD_SEQ",
        "_MODEL_PREV",
        "_MODEL_PREV_HASH",
        "_MODEL_PREV_VERSION",
        "_MODEL_PREV_PATH",
        "_MODEL_PREV2",
        "_MODEL_PREV2_HASH",
        "_MODEL_PREV2_VERSION",
        "_MODEL_PREV2_PATH",
        "_MODEL_PREV3",
        "_MODEL_PREV3_HASH",
        "_MODEL_PREV3_VERSION",
        "_MODEL_PREV3_PATH",
        "_OPCODE_AUDIT_SET",
        "_OPCODE_AUDIT_COUNT",
    ]
    return {key: getattr(classifier, key) for key in keys}


def _restore_classifier_state(state: Dict[str, Any]) -> None:
    import src.ml.classifier as classifier

    for key, value in state.items():
        setattr(classifier, key, value)


def _write_pickle(path: Path, obj: Any) -> None:
    path.write_bytes(pickle.dumps(obj, protocol=4))


def test_model_security_rollback_health_fields(tmp_path, monkeypatch) -> None:
    state = _snapshot_classifier_state()
    try:
        safe_path = tmp_path / "safe_model.pkl"
        _write_pickle(safe_path, DummyModel())
        monkeypatch.setenv("MODEL_OPCODE_MODE", "audit")
        monkeypatch.setenv("MODEL_OPCODE_SCAN", "1")
        monkeypatch.setenv("CLASSIFICATION_MODEL_VERSION", "v1")
        monkeypatch.setenv("ALLOWED_MODEL_HASHES", "")

        response = client.post(
            "/api/v1/model/reload",
            json={"path": str(safe_path)},
            headers={"X-API-Key": "test", "X-Admin-Token": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        malicious_path = tmp_path / "malicious_model.pkl"
        _write_pickle(malicious_path, len)
        monkeypatch.setenv("MODEL_OPCODE_MODE", "blacklist")
        monkeypatch.setenv("MODEL_OPCODE_SCAN", "1")

        blocked = client.post(
            "/api/v1/model/reload",
            json={"path": str(malicious_path)},
            headers={"X-API-Key": "test", "X-Admin-Token": "test"},
        )
        assert blocked.status_code == 200
        blocked_data = blocked.json()
        assert blocked_data["status"] == "opcode_blocked"
        assert blocked_data["error"]["context"]["opcode"]

        health = client.get("/api/v1/health/model", headers={"X-API-Key": "test"})
        assert health.status_code == 200
        health_data = health.json()
        assert health_data["status"] == "rollback"
        assert health_data["rollback_level"] >= 1
        assert "Disallowed pickle opcode" in (health_data.get("last_error") or "")
    finally:
        _restore_classifier_state(state)
