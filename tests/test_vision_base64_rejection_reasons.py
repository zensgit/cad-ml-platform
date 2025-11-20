import base64

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def _post(payload):
    return client.post("/api/v1/vision/analyze", json=payload)


def test_vision_base64_invalid_char_reason():
    # Contains non-base64 characters '@'
    resp = _post({"image_base64": "@@@not_base64@@@", "include_description": False})
    data = resp.json()
    assert resp.status_code == 200
    assert data["success"] is False
    assert data.get("code") == "INPUT_ERROR"
    metrics_resp = client.get("/metrics")
    if metrics_resp.status_code == 200:
        text = metrics_resp.text
        assert "vision_input_rejected_total" in text
        assert "base64_invalid_char" in text or "base64_decode_error" in text


def test_vision_base64_padding_error_reason():
    # Produce padding error by trimming a valid base64 string
    valid = base64.b64encode(b"small image bytes").decode()
    trimmed = valid.rstrip("=")[:-2]  # force incorrect padding
    resp = _post({"image_base64": trimmed, "include_description": False})
    data = resp.json()
    assert resp.status_code == 200
    assert data["success"] is False
    assert data.get("code") == "INPUT_ERROR"
    metrics_resp = client.get("/metrics")
    if metrics_resp.status_code == 200:
        text = metrics_resp.text
        assert "vision_input_rejected_total" in text
        assert "base64_padding_error" in text or "base64_decode_error" in text


def test_vision_base64_too_large_reason():
    big = b"x" * (2 * 1024 * 1024)  # 2MB
    payload = {"image_base64": base64.b64encode(big).decode(), "include_description": False}
    resp = _post(payload)
    data = resp.json()
    assert resp.status_code == 200
    assert data["success"] is False
    metrics_resp = client.get("/metrics")
    if metrics_resp.status_code == 200:
        assert "base64_too_large" in metrics_resp.text
