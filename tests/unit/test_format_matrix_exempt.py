import os
from fastapi.testclient import TestClient
from src.main import app
import io

client = TestClient(app)


def test_matrix_exempt_project():
    os.environ["FORMAT_STRICT_MODE"] = "1"
    # Create matrix file with exemption
    path = "config/format_validation_matrix.yaml"
    with open(path, "w", encoding="utf-8") as f:
        f.write("formats:\n  dxf:\n    required_tokens: ['SECTION','HEADER']\nexempt_projects: ['proj_exempt']\n")
    # Use DXF format which has more lenient validation
    # This DXF content would fail token check if not exempted
    dxf_content = b"0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nEOF" + b"X" * 100
    resp = client.post(
        "/api/v1/analyze/",
        files={"file": ("test.dxf", io.BytesIO(dxf_content), "application/octet-stream")},
        data={
            "options": '{"extract_features": true}',
            "project_id": "proj_exempt",
        },
        headers={"x-api-key": "test"},
    )
    # Exempt project should pass
    assert resp.status_code == 200, resp.text
    os.environ.pop("FORMAT_STRICT_MODE", None)
