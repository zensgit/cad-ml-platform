from fastapi.testclient import TestClient
from src.main import app
from src.utils.analysis_metrics import vector_migrate_total


client = TestClient(app)


def _counter_values(counter):
    # Extract labeled values from prometheus client Counter
    data = {}
    for sample in counter.collect()[0].samples:
        # sample.name like vector_migrate_total, labels in sample.labels
        status = sample.labels.get("status")
        if status:
            data[status] = sample.value
    return data


def test_vector_migrate_metrics_all_statuses(monkeypatch):
    # Prepare in-memory vectors with different feature versions
    from src.core import similarity
    similarity._VECTOR_STORE.clear()  # type: ignore
    similarity._VECTOR_META.clear()  # type: ignore

    similarity._VECTOR_STORE["v1_a"] = [0.1] * 7
    similarity._VECTOR_META["v1_a"] = {"feature_version": "v1"}

    similarity._VECTOR_STORE["v2_a"] = [0.2] * (7 + 5)
    similarity._VECTOR_META["v2_a"] = {"feature_version": "v2"}

    similarity._VECTOR_STORE["v3_skip"] = [0.3] * (7 + 5 + 11)
    similarity._VECTOR_META["v3_skip"] = {"feature_version": "v3"}

    # Error vector (length trigger via monkeypatch)
    similarity._VECTOR_STORE["v_err"] = list(range(999))
    similarity._VECTOR_META["v_err"] = {"feature_version": "v1"}

    # Monkeypatch upgrade_vector to raise for length 999
    from src.core.feature_extractor import FeatureExtractor as FE
    original_upgrade = FE.upgrade_vector

    def faulty(self, existing):  # type: ignore
        if len(existing) == 999:
            raise Exception("upgrade_failed")
        return original_upgrade(self, existing)

    monkeypatch.setattr(FE, "upgrade_vector", faulty)

    # Dry-run migration (should yield dry_run + skipped + not_found + error)
    dry_payload = {
        "ids": ["v1_a", "v2_a", "v3_skip", "missing_id", "v_err"],
        "to_version": "v3",
        "dry_run": True,
    }
    r = client.post("/api/v1/vectors/migrate", json=dry_payload, headers={"x-api-key": "test"})
    assert r.status_code == 200
    data = r.json()
    statuses = {item["status"] for item in data["items"]}
    assert "dry_run" in statuses
    assert "skipped" in statuses
    assert "not_found" in statuses
    assert "error" in statuses

    # Real migration (migrated should appear)
    real_payload = {
        "ids": ["v1_a", "v2_a"],
        "to_version": "v3",
        "dry_run": False,
    }
    r2 = client.post("/api/v1/vectors/migrate", json=real_payload, headers={"x-api-key": "test"})
    assert r2.status_code == 200
    data2 = r2.json()
    assert any(item["status"] == "migrated" for item in data2["items"])

    # Collect counter values
    values = _counter_values(vector_migrate_total)
    # Ensure each status had at least one increment
    for status in ["dry_run", "skipped", "not_found", "error", "migrated"]:
        assert status in values and values[status] > 0, f"Missing counter for {status}" 


def test_vector_migrate_invalid_version():
    payload = {"ids": ["any"], "to_version": "v999", "dry_run": True}
    r = client.post("/api/v1/vectors/migrate", json=payload, headers={"x-api-key": "test"})
    assert r.status_code == 422
    data = r.json()
    detail = data["detail"]
    assert detail["code"] == "INPUT_VALIDATION_FAILED"
    assert detail["stage"] == "vector_migrate"
