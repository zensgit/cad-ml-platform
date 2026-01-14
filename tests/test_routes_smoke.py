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


def test_ocr_extract_base64_route_exists():
    payload = {
        "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMBAkYp9V0AAAAASUVORK5CYII="
    }
    resp = client.post("/api/v1/ocr/extract-base64", json=payload)
    assert resp.status_code in (200, 400, 415, 500)


def test_drawing_recognize_route_exists():
    # Send a tiny fake image payload
    files = {"file": ("test.png", b"fake_image_bytes", "image/png")}
    resp = client.post("/api/v1/drawing/recognize?provider=auto", files=files)
    # Endpoint should exist and not be 404 due to routing
    assert resp.status_code in (200, 400, 415, 429, 500)


def test_drawing_fields_route_exists():
    resp = client.get("/api/v1/drawing/fields")
    assert resp.status_code == 200


def test_drawing_recognize_base64_route_exists():
    payload = {
        "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMBAkYp9V0AAAAASUVORK5CYII="
    }
    resp = client.post("/api/v1/drawing/recognize-base64", json=payload)
    assert resp.status_code == 200


def test_ocr_providers_route_exists():
    resp = client.get("/api/v1/ocr/providers")
    assert resp.status_code == 200
    data = resp.json()
    assert "providers" in data


def test_ocr_health_route_exists():
    resp = client.get("/api/v1/ocr/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "providers" in data


def test_drawing_providers_route_exists():
    resp = client.get("/api/v1/drawing/providers")
    assert resp.status_code == 200
    data = resp.json()
    assert "providers" in data


def test_drawing_health_route_exists():
    resp = client.get("/api/v1/drawing/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
