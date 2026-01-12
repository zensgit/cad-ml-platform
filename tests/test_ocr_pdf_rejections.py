import re

from fastapi.testclient import TestClient

from src.core.errors import ErrorCode
from src.main import app

client = TestClient(app)


def _assert_rejection_metric(metrics_text: str, reason: str) -> None:
    pattern = rf'ocr_input_rejected_total(_total)?\{{[^}}]*reason="{reason}"'
    assert re.search(pattern, metrics_text)


def _make_pdf(pages: int, forbidden: bool = False) -> bytes:
    # Minimal synthetic PDF generator
    header = b"%PDF-1.4\n"
    body = b""
    for i in range(pages):
        body += f"%Page {i}\n".encode()
    if forbidden:
        body += b"/JavaScript"  # forbidden token
    return header + body + b"%%EOF"


def test_ocr_pdf_pages_exceed(metrics_text):
    pdf_bytes = _make_pdf(25)  # exceeds default 20
    files = {"file": ("large.pdf", pdf_bytes, "application/pdf")}
    resp = client.post("/api/v1/ocr/extract", files=files)
    data = resp.json()
    assert resp.status_code == 200
    assert data["success"] is False
    assert data.get("code") == ErrorCode.INPUT_ERROR
    assert "page count" in (data.get("error") or "").lower()
    metrics = metrics_text(client)
    if metrics:
        _assert_rejection_metric(metrics, "pdf_pages_exceed")


def test_ocr_pdf_forbidden_token(metrics_text):
    pdf_bytes = _make_pdf(2, forbidden=True)
    files = {"file": ("bad.pdf", pdf_bytes, "application/pdf")}
    resp = client.post("/api/v1/ocr/extract", files=files)
    data = resp.json()
    assert resp.status_code == 200
    assert data["success"] is False
    assert data.get("code") == ErrorCode.INPUT_ERROR
    assert "forbidden token" in (data.get("error") or "").lower()
    metrics = metrics_text(client)
    if metrics:
        _assert_rejection_metric(metrics, "pdf_forbidden_token")
