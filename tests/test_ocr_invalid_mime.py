import re

from fastapi.testclient import TestClient

from src.core.errors import ErrorCode
from src.main import app

client = TestClient(app)


def test_ocr_invalid_mime(metrics_text):
    # Upload a fake text file disguised as image - validator should reject
    files = {"file": ("fake.txt", b"not_an_image", "text/plain")}
    resp = client.post("/api/v1/ocr/extract", files=files)
    data = resp.json()
    assert resp.status_code == 200
    assert data["success"] is False
    assert data.get("code") == ErrorCode.INPUT_ERROR
    assert "mime" in (data.get("error") or "").lower()
    metrics = metrics_text(client)
    if metrics:
        pattern = r'ocr_input_rejected_total(_total)?\{[^}]*reason="invalid_mime"'
        assert re.search(pattern, metrics)
