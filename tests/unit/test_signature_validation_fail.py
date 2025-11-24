from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_signature_validation_failure(monkeypatch):
    # Provide content that will fail STEP signature (wrong header)
    fake_content = b"XXXX-INVALID-STEP"  # not starting with ISO-10303-21
    files = {"file": ("test.step", fake_content, "application/octet-stream")}
    r = client.post(
        "/api/v1/",
        data={"options": '{"extract_features": false}'},
        files=files,
        headers={"api-key": "test"},
    )
    assert r.status_code == 415
    body = r.json()
    assert "detail" in body
    detail = body["detail"]
    assert isinstance(detail, dict)
    assert detail.get("code") == "INPUT_FORMAT_INVALID"
    assert "signature_prefix" in detail
    assert "expected_signature" in detail
