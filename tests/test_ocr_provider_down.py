from fastapi.testclient import TestClient

from src.api.v1.ocr import get_manager
from src.core.errors import ErrorCode
from src.main import app

client = TestClient(app)


def test_ocr_provider_down(metrics_text):
    manager = get_manager()
    # Remove a provider temporarily
    if "paddle" in manager.providers:
        del manager.providers["paddle"]
    files = {"file": ("fake.png", b"fakeimgbytes", "image/png")}
    resp = client.post("/api/v1/ocr/extract?provider=paddle", files=files)
    data = resp.json()
    assert resp.status_code == 200
    assert data["success"] is False
    # Expect PROVIDER_DOWN when provider missing
    assert data.get("code") == ErrorCode.PROVIDER_DOWN
    metrics = metrics_text(client)
    if metrics:
        # Expect a labeled counter increment: ocr_errors_total{code="provider_down"}
        assert "ocr_errors_total" in metrics
        assert "provider_down" in metrics
