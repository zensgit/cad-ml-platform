import io
import os

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_ml_classifier_fallback_model_unavailable():
    # Ensure model path does not exist
    os.environ["CLASSIFICATION_MODEL_PATH"] = "models/nonexistent.pkl"
    os.environ["CLASSIFICATION_MODEL_VERSION"] = "v0"
    content = b"ISO-10303-21;HEADER;ENDSEC;DATA;ENDSEC;EOF" + b"X" * 40
    resp = client.post(
        "/api/v1/analyze/",
        files={"file": ("sample.step", io.BytesIO(content), "application/step")},
        data={"options": '{"extract_features": true, "classify_parts": true}'},
        headers={"x-api-key": "test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    cls = data["results"].get("classification")
    assert cls is not None
    assert "model_version" in cls
    assert cls["model_version"] in {"model_unavailable", "ml_error", "inference_error", "v0"}
