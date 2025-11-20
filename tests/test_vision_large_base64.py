from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_vision_large_base64_rejected():
    # Create >1MB decoded payload (base64 of ~1.2MB raw bytes)
    raw = b"x" * (1024 * 1200)  # ~1.17MB
    import base64

    payload = {
        "image_base64": base64.b64encode(raw).decode(),
        "include_description": False,
        "include_ocr": False,
    }
    resp = client.post("/api/v1/vision/analyze", json=payload)
    data = resp.json()
    assert resp.status_code == 200
    assert data["success"] is False
    assert data.get("code") == "INPUT_ERROR"
