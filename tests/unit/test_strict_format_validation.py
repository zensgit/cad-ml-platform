import io
import os

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_step_deep_validation_fail_strict_mode():
    os.environ["FORMAT_STRICT_MODE"] = "1"
    # Missing ISO-10303-21 header
    bad_step = b"HEADER;ENDSEC;DATA;ENDSEC;" + b"X" * 50
    resp = client.post(
        "/api/v1/analyze/",
        files={"file": ("bad.step", io.BytesIO(bad_step), "application/step")},
        data={"options": '{"extract_features": true}'},
        headers={"x-api-key": "test"},
    )
    assert resp.status_code == 415
    body = resp.json()
    assert (
        body.get("code") == "INPUT_FORMAT_INVALID"
        or body.get("detail", {}).get("code") == "INPUT_FORMAT_INVALID"
    )


def test_step_deep_validation_pass_strict_mode():
    os.environ["FORMAT_STRICT_MODE"] = "1"
    good_step = b"ISO-10303-21;HEADER;ENDSEC;DATA;ENDSEC;EOF" + b"X" * 50
    resp = client.post(
        "/api/v1/analyze/",
        files={"file": ("good.step", io.BytesIO(good_step), "application/step")},
        data={"options": '{"extract_features": true}'},
        headers={"x-api-key": "test"},
    )
    assert resp.status_code == 200, resp.text
