from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_step_failure_graceful():
    # Provide invalid STEP bytes to trigger fallback path
    file = ("bad.step", b"NOT_A_STEP_FILE", "application/octet-stream")
    r = client.post(
        "/api/v1/analyze",
        files={"file": file},
        data={"options": '{"extract_features": true, "classify_parts": false}'},
        headers={"X-API-Key": "test"},
    )
    assert r.status_code == 200  # should not crash
    data = r.json()
    assert data["file_format"].lower() in ("step", "stp")
    # Fallback complexity present
    assert "statistics" in data["results"]
    # Features returned even if no entities parsed
    assert "features" in data["results"]

