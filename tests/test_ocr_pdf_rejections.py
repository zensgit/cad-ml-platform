from fastapi.testclient import TestClient
from src.main import app
from src.core.errors import ErrorCode

client = TestClient(app)


def _make_pdf(pages: int, forbidden: bool = False) -> bytes:
    # Minimal synthetic PDF generator
    header = b"%PDF-1.4\n"
    body = b""
    for i in range(pages):
        body += f"%Page {i}\n".encode()
    if forbidden:
        body += b"/JavaScript"  # forbidden token
    return header + body + b"%%EOF"


def test_ocr_pdf_pages_exceed():
    pdf_bytes = _make_pdf(25)  # exceeds default 20
    files = {"file": ("large.pdf", pdf_bytes, "application/pdf")}
    resp = client.post("/api/v1/ocr/extract", files=files)
    data = resp.json()
    assert resp.status_code == 200
    assert data["success"] is False
    assert data.get("code") == ErrorCode.INPUT_ERROR
    metrics_resp = client.get("/metrics")
    if metrics_resp.status_code == 200:
        assert "pdf_pages_exceed" in metrics_resp.text


def test_ocr_pdf_forbidden_token():
    pdf_bytes = _make_pdf(2, forbidden=True)
    files = {"file": ("bad.pdf", pdf_bytes, "application/pdf")}
    resp = client.post("/api/v1/ocr/extract", files=files)
    data = resp.json()
    assert resp.status_code == 200
    assert data["success"] is False
    assert data.get("code") == ErrorCode.INPUT_ERROR
    metrics_resp = client.get("/metrics")
    if metrics_resp.status_code == 200:
        assert "pdf_forbidden_token" in metrics_resp.text
