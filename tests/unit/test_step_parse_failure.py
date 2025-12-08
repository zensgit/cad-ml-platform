from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_step_failure_graceful():
    # Provide invalid STEP bytes - with signature validation enabled,
    # this should return 415 for invalid format
    file = ("bad.step", b"NOT_A_STEP_FILE", "application/octet-stream")
    r = client.post(
        "/api/v1/analyze/",
        files={"file": file},
        data={"options": '{"extract_features": true, "classify_parts": false}'},
        headers={"X-API-Key": "test"},
    )
    # Signature validation catches invalid STEP format
    assert r.status_code == 415
    data = r.json()
    assert "detail" in data
    detail = data["detail"]
    assert detail.get("code") == "INPUT_FORMAT_INVALID"

