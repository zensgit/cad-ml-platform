import io
import os

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_step_signature_validation_success(monkeypatch):
    # Minimal STEP header
    content = b"ISO-10303-21;HEADER;ENDSEC;DATA;ENDSEC;EOF" + b"X" * 50
    resp = client.post(
        "/api/v1/analyze/",
        files={"file": ("part.step", io.BytesIO(content), "application/step")},
        data={"options": '{"extract_features": true}', "material": "steel"},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )
    assert resp.status_code == 200, resp.text


def test_step_signature_validation_fail(monkeypatch):
    # Wrong header for STEP (simulate mismatch)
    bad_content = b"NOTSTEP" + b"Y" * 120
    resp = client.post(
        "/api/v1/analyze/",
        files={"file": ("bad.step", io.BytesIO(bad_content), "application/step")},
        data={"options": '{"extract_features": true}'},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )
    # Expect 415
    assert resp.status_code == 415, resp.text
    assert "Signature validation failed" in resp.text
