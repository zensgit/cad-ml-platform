from datetime import datetime

from fastapi.testclient import TestClient

from src.core.feature_extractor import SLOTS_V1, SLOTS_V2, SLOTS_V3, SLOTS_V4
from src.core.similarity import _VECTOR_META, _VECTOR_STORE  # type: ignore
from src.main import app

client = TestClient(app)


def _register_vec(vid: str, version: str):
    length = len(SLOTS_V1)
    if version in {"v2", "v3", "v4"}:
        length += len(SLOTS_V2)
    if version in {"v3", "v4"}:
        length += len(SLOTS_V3)
    if version == "v4":
        length += len(SLOTS_V4)
    _VECTOR_STORE[vid] = [0.1] * length
    _VECTOR_META[vid] = {"feature_version": version, "material": "steel"}


def test_migrate_v1_to_v4_dry_run():
    _register_vec("v1_to_v4", "v1")
    resp = client.post(
        "/api/v1/vectors/migrate",
        json={"ids": ["v1_to_v4"], "to_version": "v4", "dry_run": True},
        headers={"x-api-key": "test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    item = data["items"][0]
    assert item["status"] == "dry_run"
    assert item["dimension_before"] == len(SLOTS_V1)
    target_len = len(SLOTS_V1) + len(SLOTS_V2) + len(SLOTS_V3) + len(SLOTS_V4)
    assert item["dimension_after"] == target_len
    # Original vector should remain unchanged (dry-run)
    assert len(_VECTOR_STORE["v1_to_v4"]) == len(SLOTS_V1)


def test_migrate_v3_to_v4_actual():
    _register_vec("v3_to_v4", "v3")
    resp = client.post(
        "/api/v1/vectors/migrate",
        json={"ids": ["v3_to_v4"], "to_version": "v4", "dry_run": False},
        headers={"x-api-key": "test"},
    )
    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["status"] == "migrated"
    assert len(_VECTOR_STORE["v3_to_v4"]) == len(SLOTS_V1) + len(SLOTS_V2) + len(SLOTS_V3) + len(
        SLOTS_V4
    )
    assert _VECTOR_META["v3_to_v4"]["feature_version"] == "v4"


def test_migrate_v4_to_v2_downgraded():
    _register_vec("v4_to_v2", "v4")
    resp = client.post(
        "/api/v1/vectors/migrate",
        json={"ids": ["v4_to_v2"], "to_version": "v2", "dry_run": False},
        headers={"x-api-key": "test"},
    )
    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["status"] == "downgraded"
    assert len(_VECTOR_STORE["v4_to_v2"]) == len(SLOTS_V1) + len(SLOTS_V2)
    assert _VECTOR_META["v4_to_v2"]["feature_version"] == "v2"


def test_migrate_invalid_target_version():
    _register_vec("invalid_target", "v1")
    resp = client.post(
        "/api/v1/vectors/migrate",
        json={"ids": ["invalid_target"], "to_version": "v999", "dry_run": True},
        headers={"x-api-key": "test"},
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "INPUT_VALIDATION_FAILED"
    assert detail["stage"] == "vector_migrate"


def test_migrate_summary_endpoint():
    # Use previously registered vectors; call summary
    resp = client.get(
        "/api/v1/vectors/migrate/summary",
        headers={"x-api-key": "test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "counts" in data
    assert "total_migrations" in data
    assert isinstance(data["statuses"], list)
