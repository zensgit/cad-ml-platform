from fastapi.testclient import TestClient
from src.main import app
from src.core.errors import ErrorCode
from src.api.v1.ocr import get_manager

client = TestClient(app)


def test_ocr_provider_down():
    manager = get_manager()
    # Remove a provider temporarily
    if "paddle" in manager.providers:
        del manager.providers["paddle"]
    files = {"file": ("fake.png", b"fakeimgbytes", "image/png")}
    resp = client.post("/api/v1/ocr/extract?provider=paddle", files=files)
    data = resp.json()
    assert resp.status_code == 200
    assert data["success"] is False
    assert data.get("code") in (ErrorCode.INTERNAL_ERROR, ErrorCode.INPUT_ERROR, ErrorCode.PROVIDER_DOWN)
    metrics_resp = client.get("/metrics")
    if metrics_resp.status_code == 200:
        metrics = metrics_resp.text
        assert "provider_down" in metrics or data.get("code") == ErrorCode.PROVIDER_DOWN
