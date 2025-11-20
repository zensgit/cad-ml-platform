import base64

from fastapi.testclient import TestClient

from src.core.errors import ErrorCode
from src.main import app

client = TestClient(app)


def test_ocr_extract_empty_file():
    # Empty base64 should trigger input validation failure inside OCR/vision path
    payload = {
        "image_base64": base64.b64encode(b"").decode(),
        "include_description": False,
        "include_ocr": True,
        "ocr_provider": "auto",
    }
    resp = client.post("/api/v1/vision/analyze", json=payload)
    data = resp.json()
    assert resp.status_code == 200
    assert data["success"] is False
    assert data.get("code") == ErrorCode.INPUT_ERROR


def test_ocr_extract_invalid_base64():
    payload = {
        "image_base64": "@@@not_base64@@@",
        "include_description": False,
        "include_ocr": True,
        "ocr_provider": "auto",
    }
    resp = client.post("/api/v1/vision/analyze", json=payload)
    data = resp.json()
    assert resp.status_code == 200
    assert data["success"] is False
    assert data.get("code") == ErrorCode.INPUT_ERROR


def test_ocr_provider_not_found():
    files = {"file": ("fake.png", b"fakeimgbytes", "image/png")}
    resp = client.post("/api/v1/ocr/extract?provider=invalid", files=files)
    data = resp.json()
    assert resp.status_code == 200
    assert data["success"] is False
    # Explicit invalid provider should surface PROVIDER_DOWN in unified error model
    assert data.get("code") in (
        ErrorCode.INTERNAL_ERROR,
        ErrorCode.INPUT_ERROR,
        ErrorCode.PROVIDER_DOWN,
    )
