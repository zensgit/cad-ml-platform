import pytest
from fastapi.testclient import TestClient

from src.core.errors import ErrorCode
from src.main import app

client = TestClient(app)


@pytest.mark.skip(reason="MIME validation not yet implemented in validate_and_read()")
def test_ocr_invalid_mime():
    # Upload a fake text file disguised as image - validator should reject
    # TODO: Implement MIME validation in src/security/input_validator.py
    files = {"file": ("fake.txt", b"not_an_image", "text/plain")}
    resp = client.post("/api/v1/ocr/extract", files=files)
    data = resp.json()
    assert resp.status_code == 200
    assert data["success"] is False
    assert data.get("code") == ErrorCode.INPUT_ERROR
