"""Basic smoke tests for normalized API routes.

Validates that key endpoints are mounted under /api/v1 without duplication.
"""

from fastapi.testclient import TestClient
from src.main import app


client = TestClient(app)


def test_vision_health_route():
    resp = client.get("/api/v1/vision/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "healthy"
    assert "provider" in data


def test_ocr_extract_route_exists():
    # Send a tiny fake image payload
    files = {"file": ("test.png", b"fake_image_bytes", "image/png")}
    resp = client.post("/api/v1/ocr/extract?provider=auto", files=files)
    # Endpoint should exist and not be 404 due to routing
    assert resp.status_code in (200, 400, 415, 500)
