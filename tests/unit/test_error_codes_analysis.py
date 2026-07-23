import json
import os
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

VALID_LINE_DXF = (
    b"0\nSECTION\n2\nENTITIES\n"
    b"0\nLINE\n8\n0\n10\n0\n20\n0\n30\n0\n11\n1\n21\n0\n31\n0\n"
    b"0\nENDSEC\n0\nEOF\n"
)


def test_options_json_parse_error():
    r = client.post(
        "/api/v1/analyze",
        files={"file": ("f.dxf", b"123", "application/octet-stream")},
        data={"options": "{bad json}"},
        headers={"X-API-Key": "test"},
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "JSON_PARSE_ERROR"
    assert detail["source"] == "input"


def test_entity_limit_violation():
    # Force limit 0 with a valid one-entity DXF so the test exercises
    # entity-limit validation instead of parser stub behavior.
    with patch.dict(os.environ, {"ANALYSIS_MAX_ENTITIES": "0"}):
        r = client.post(
            "/api/v1/analyze",
            files={"file": ("g.dxf", VALID_LINE_DXF, "application/octet-stream")},
            data={"options": '{"extract_features": false, "classify_parts": false}'},
            headers={"X-API-Key": "test"},
        )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["code"] == "VALIDATION_FAILED"


def test_file_size_exceeded():
    big_content = b"0" * 50000  # ~50KB
    with patch.dict(os.environ, {"ANALYSIS_MAX_FILE_MB": "0.00001"}):
        r = client.post(
            "/api/v1/analyze",
            files={"file": ("big.dxf", big_content, "application/octet-stream")},
            data={"options": '{"extract_features": false}'},
            headers={"X-API-Key": "test"},
        )
    assert r.status_code == 413
    detail = r.json()["detail"]
    assert detail["code"] == "INPUT_SIZE_EXCEEDED"


def test_analyze_rejects_raw_dwg_fail_closed():
    r = client.post(
        "/api/v1/analyze",
        files={
            "file": (
                "raw.dwg",
                b"AC1018\x00binary-dwg-placeholder",
                "image/vnd.dwg",
            )
        },
        data={"options": '{"extract_features": false, "classify_parts": false}'},
        headers={"X-API-Key": "test"},
    )
    assert r.status_code == 415
    detail = r.json()["detail"]
    assert detail["code"] == "UNSUPPORTED_INPUT_DWG"
    assert detail["source"] == "input"
    assert detail["context"]["format"] == "dwg"


def test_analyze_rejects_empty_dxf_parser_stub_fail_closed():
    r = client.post(
        "/api/v1/analyze",
        files={"file": ("broken.dxf", b"456", "application/octet-stream")},
        data={"options": '{"extract_features": false, "classify_parts": false}'},
        headers={"X-API-Key": "test"},
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["code"] == "PARSE_FAILED"
    assert detail["source"] == "system"
    assert detail["context"]["format"] == "dxf"
