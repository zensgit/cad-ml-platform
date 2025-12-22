from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def _headers():
    return {"X-API-Key": "test"}


def test_drift_baseline_export_empty():
    response = client.post("/api/v1/analyze/drift/baseline/export", headers=_headers())
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"empty", "ok"}
    snapshot = data["snapshot"]
    assert snapshot["material_baseline"] == []
    assert snapshot["prediction_baseline"] == []
    assert "exported_at" in snapshot


def test_drift_baseline_import_roundtrip():
    payload = {
        "material_baseline": ["steel", "aluminum"],
        "prediction_baseline": ["bracket"],
        "material_baseline_ts": 1700000000,
        "prediction_baseline_ts": 1700001234,
    }
    import_resp = client.post(
        "/api/v1/analyze/drift/baseline/import",
        headers=_headers(),
        json=payload,
    )
    assert import_resp.status_code == 200
    import_data = import_resp.json()
    assert import_data["status"] == "ok"
    assert import_data["imported_materials"] is True
    assert import_data["imported_predictions"] is True

    export_resp = client.post("/api/v1/analyze/drift/baseline/export", headers=_headers())
    assert export_resp.status_code == 200
    export_data = export_resp.json()["snapshot"]
    assert export_data["material_baseline"] == payload["material_baseline"]
    assert export_data["prediction_baseline"] == payload["prediction_baseline"]
    assert export_data["material_baseline_ts"] == payload["material_baseline_ts"]
    assert export_data["prediction_baseline_ts"] == payload["prediction_baseline_ts"]
