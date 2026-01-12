import base64

from fastapi.testclient import TestClient

from src.core.errors import ErrorCode  # noqa: F401 (referenced if extending tests)
from src.main import app

client = TestClient(app)
_SAMPLE_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z/C/HwAFgwJ/lb9a0QAAAABJRU5ErkJggg=="
)


def _metrics_text_if_enabled() -> str | None:
    response = client.get("/metrics")
    if response.status_code != 200:
        return None
    if "app_metrics_disabled" in response.text:
        return None
    return response.text


def test_health_contains_runtime_info():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "runtime" in data
    assert "python_version" in data["runtime"]
    assert "metrics_enabled" in data["runtime"]
    # error_rate_ema structure present
    assert "error_rate_ema" in data["runtime"]
    assert "ocr" in data["runtime"]["error_rate_ema"]
    assert "vision" in data["runtime"]["error_rate_ema"]

    # Touch endpoints to update EMA values
    client.post(
        "/api/v1/vision/analyze", json={"image_base64": "aGVsbG8=", "include_description": False}
    )
    files = {"file": ("fake.png", _SAMPLE_PNG_BYTES, "image/png")}
    client.post("/api/v1/ocr/extract", files=files)
    data2 = client.get("/health").json()
    assert "ocr" in data2["runtime"]["error_rate_ema"]
    assert "vision" in data2["runtime"]["error_rate_ema"]


def test_metrics_has_vision_and_ocr_counters():
    payload = {"image_base64": "aGVsbG8=", "include_description": False, "include_ocr": False}
    client.post("/api/v1/vision/analyze", json=payload)
    # OCR extract with a valid PNG to ensure provider metrics register
    files = {"file": ("fake.png", _SAMPLE_PNG_BYTES, "image/png")}
    client.post("/api/v1/ocr/extract", files=files)
    text = _metrics_text_if_enabled()
    if text is None:
        return
    assert "vision_requests_total" in text
    # Image size histogram should appear once we observed a payload
    assert "vision_image_size_bytes" in text
    # Expect at least one success or error line
    lines = [l for l in text.splitlines() if l.startswith("vision_requests_total")]
    assert any('status="success"' in l or 'status="error"' in l for l in lines)
    # OCR metrics should be present - either requests or input_rejected
    # The file may succeed or fail depending on implementation, so check both
    ocr_metric_present = "ocr_requests_total" in text or "ocr_input_rejected_total" in text
    assert ocr_metric_present, "Expected at least one OCR metric to be present"

    # Provider loaded gauge should be present for at least one provider (e.g., paddle)
    assert "ocr_model_loaded" in text
    # We can't guarantee provider name, but ensure there's a sample line with provider label
    gauge_lines = [
        l for l in text.splitlines() if l.startswith("ocr_model_loaded") and "provider=" in l
    ]
    assert gauge_lines


def test_metrics_rejected_counter_for_large_base64():
    import base64

    raw = b"x" * (1024 * 1200)
    payload = {
        "image_base64": base64.b64encode(raw).decode(),
        "include_description": False,
        "include_ocr": False,
    }
    client.post("/api/v1/vision/analyze", json=payload)
    text = _metrics_text_if_enabled()
    if text is None:
        return
    assert "vision_input_rejected_total" in text
    assert "base64_too_large" in text
