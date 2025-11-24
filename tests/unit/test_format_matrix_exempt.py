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
        f.write("formats:\n  step:\n    required_tokens: ['ISO-10303-21','HEADER']\nexempt_projects: ['proj_exempt']\n")
    bad_step = b"HEADER;ENDSEC;DATA;ENDSEC;" + b"X" * 50  # missing ISO-10303-21 token
    resp = client.post(
        "/api/v1/analyze/",
        files={"file": ("bad.step", io.BytesIO(bad_step), "application/step")},
        data={
            "options": '{"extract_features": true}',
            "project_id": "proj_exempt",
        },
        headers={"x-api-key": "test"},
    )
    # Exempt project should pass despite missing token
    assert resp.status_code == 200, resp.text
