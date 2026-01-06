"""Vision endpoint tests - MVP happy path validation.

Tests:
1. /api/v1/vision/analyze with stub provider
2. /api/v1/vision/health check
3. Error handling for invalid inputs
"""

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

# ========== Test Fixtures ==========


def _encode_image_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@pytest.fixture(autouse=True)
def _force_stub_provider(monkeypatch):
    """Force stub provider for deterministic tests.

    Other test modules may set external provider API keys (e.g. OPENAI_API_KEY),
    which would make the vision router auto-select a real provider and attempt
    network calls.
    """
    monkeypatch.setenv("VISION_PROVIDER", "deepseek_stub")
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "DASHSCOPE_API_KEY",
        "QWEN_API_KEY",
        "ZHIPUAI_API_KEY",
        "GLM_API_KEY",
        "ARK_API_KEY",
        "VOLCENGINE_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    from src.api.v1.vision import reset_vision_manager

    reset_vision_manager()
    yield
    reset_vision_manager()


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Return sample image bytes (1x1 PNG)."""
    # Minimal 1x1 PNG (black pixel)
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00:~\x9bU\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    return png_bytes


@pytest.fixture
def sample_image_base64(sample_image_bytes) -> str:
    """Return sample image as base64 string."""
    return base64.b64encode(sample_image_bytes).decode("utf-8")


# ========== Vision Endpoint Tests ==========


def test_vision_analyze_with_base64_happy_path(sample_image_base64):
    """
    Test /api/v1/vision/analyze with valid base64 image.

    Expected behavior:
    - Returns success=True
    - Description present from stub provider
    - No OCR (manager not connected yet)
    - Processing time > 0
    """
    from fastapi import FastAPI

    from src.api.v1.vision import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/vision")
    client = TestClient(app)

    # Prepare request
    request_data = {
        "image_base64": sample_image_base64,
        "include_description": True,
        "include_ocr": False,  # OCR not yet connected
        "ocr_provider": "auto",
    }

    # Make request
    response = client.post("/api/v1/vision/analyze", json=request_data)

    # Assertions
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["provider"] == "deepseek_stub"
    assert data["description"] is not None
    assert "cylindrical part" in data["description"]["summary"].lower()
    assert len(data["description"]["details"]) > 0
    assert 0.0 < data["description"]["confidence"] <= 1.0
    assert data["ocr"] is None  # Not yet connected
    assert data["processing_time_ms"] > 0


def test_vision_analyze_includes_cad_stats(sample_image_base64):
    """Test /api/v1/vision/analyze returns cad_feature_stats when requested."""
    from fastapi import FastAPI

    from src.api.v1.vision import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/vision")
    client = TestClient(app)

    request_data = {
        "image_base64": sample_image_base64,
        "include_description": True,
        "include_ocr": False,
        "include_cad_stats": True,
    }

    response = client.post("/api/v1/vision/analyze", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    stats = data.get("cad_feature_stats")
    assert stats is not None
    assert stats["line_count"] == 0
    assert stats["circle_count"] == 0
    assert stats["arc_count"] == 0
    assert sum(stats["line_angle_bins"].values()) == 0
    assert stats["line_angle_avg"] is None
    assert stats["arc_sweep_avg"] is None


def test_vision_analyze_invalid_cad_threshold_key(sample_image_base64):
    """Test /api/v1/vision/analyze rejects unknown CAD threshold keys."""
    from fastapi import FastAPI

    from src.api.v1.vision import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/vision")
    client = TestClient(app)

    request_data = {
        "image_base64": sample_image_base64,
        "include_description": True,
        "include_ocr": False,
        "include_cad_stats": True,
        "cad_feature_thresholds": {"unknown_key": 1.0},
    }

    response = client.post("/api/v1/vision/analyze", json=request_data)

    assert response.status_code == 422


def test_vision_analyze_invalid_cad_threshold_value(sample_image_base64):
    """Test /api/v1/vision/analyze rejects invalid threshold values."""
    from fastapi import FastAPI

    from src.api.v1.vision import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/vision")
    client = TestClient(app)

    request_data = {
        "image_base64": sample_image_base64,
        "include_description": True,
        "include_ocr": False,
        "include_cad_stats": True,
        "cad_feature_thresholds": {"line_aspect": 0},
    }

    response = client.post("/api/v1/vision/analyze", json=request_data)

    assert response.status_code == 422


def test_vision_analyze_thresholds_change_stats(sample_image_base64):
    """Test /api/v1/vision/analyze returns different stats with thresholds."""
    from fastapi import FastAPI

    from src.api.v1.vision import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/vision")
    client = TestClient(app)

    image = Image.new("L", (120, 80), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((10, 10, 110, 10), fill=0, width=3)
    line_image_base64 = _encode_image_base64(image)

    request_base = {
        "image_base64": line_image_base64,
        "include_description": True,
        "include_ocr": False,
        "include_cad_stats": True,
    }

    default_response = client.post("/api/v1/vision/analyze", json=request_base)
    assert default_response.status_code == 200
    default_stats = default_response.json()["cad_feature_stats"]

    strict_response = client.post(
        "/api/v1/vision/analyze",
        json={
            **request_base,
            "cad_feature_thresholds": {"min_area": 1000000},
        },
    )
    assert strict_response.status_code == 200
    strict_stats = strict_response.json()["cad_feature_stats"]

    assert default_stats["line_count"] >= 1
    assert strict_stats["line_count"] == 0
    assert strict_stats["circle_count"] == 0
    assert strict_stats["arc_count"] == 0
    assert strict_stats != default_stats


def test_vision_analyze_missing_image_error():
    """
    Test /api/v1/vision/analyze with missing image data.

    Expected behavior:
    - Returns HTTP 400
    - Error message indicates missing input
    """
    from fastapi import FastAPI

    from src.api.v1.vision import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/vision")
    client = TestClient(app)

    # Request with no image
    request_data = {"include_description": True, "include_ocr": False}

    response = client.post("/api/v1/vision/analyze", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data.get("code") == "INPUT_ERROR"


def test_vision_analyze_invalid_base64_error():
    """
    Test /api/v1/vision/analyze with invalid base64 data.

    Expected behavior:
    - Returns HTTP 400
    - Error message indicates invalid base64
    """
    from fastapi import FastAPI

    from src.api.v1.vision import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/vision")
    client = TestClient(app)

    # Invalid base64
    request_data = {"image_base64": "this-is-not-valid-base64!!!", "include_description": True}

    response = client.post("/api/v1/vision/analyze", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data.get("code") == "INPUT_ERROR"


def test_vision_health_check():
    """
    Test /api/v1/vision/health endpoint.

    Expected behavior:
    - Returns 200
    - Status is healthy
    - Provider name is deepseek_stub
    - OCR enabled is true (OCRManager connected in Phase 2)
    """
    from fastapi import FastAPI

    from src.api.v1.vision import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/vision")
    client = TestClient(app)

    response = client.get("/api/v1/vision/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["provider"] == "deepseek_stub"
    assert data["ocr_enabled"] is True  # OCRManager connected in Phase 2


# ========== Stub Provider Direct Tests ==========


@pytest.mark.asyncio
async def test_stub_provider_direct(sample_image_bytes):
    """
    Test DeepSeekStubProvider directly (unit test).

    Validates:
    - Fixed description generation
    - Simulated latency
    - Confidence score
    """
    from src.core.vision.providers import create_stub_provider

    provider = create_stub_provider(simulate_latency_ms=10.0)

    # Analyze image
    result = await provider.analyze_image(image_data=sample_image_bytes, include_description=True)

    # Assertions
    assert result.summary is not None
    assert "cylindrical" in result.summary.lower() or "mechanical" in result.summary.lower()
    assert len(result.details) > 0
    assert 0.0 < result.confidence <= 1.0


@pytest.mark.asyncio
async def test_stub_provider_no_description(sample_image_bytes):
    """
    Test stub provider with include_description=False.

    Expected:
    - Minimal description
    - Confidence still present
    """
    from src.core.vision.providers import create_stub_provider

    provider = create_stub_provider(simulate_latency_ms=0)

    result = await provider.analyze_image(image_data=sample_image_bytes, include_description=False)

    # Should return minimal description
    assert result.summary is not None
    assert len(result.details) == 0
    assert result.confidence == 1.0


@pytest.mark.asyncio
async def test_stub_provider_empty_image_error():
    """
    Test stub provider with empty image data.

    Expected:
    - Raises ValueError
    """
    from src.core.vision.providers import create_stub_provider

    provider = create_stub_provider()

    with pytest.raises(ValueError, match="cannot be empty"):
        await provider.analyze_image(image_data=b"", include_description=True)  # Empty bytes


# ========== Vision Manager Tests ==========


@pytest.mark.asyncio
async def test_vision_manager_without_ocr(sample_image_base64):
    """
    Test VisionManager end-to-end without OCR integration.

    Validates:
    - Request → Manager → Response flow
    - Description present, OCR absent
    - Processing time tracked
    """
    from src.core.vision import VisionAnalyzeRequest, VisionManager, create_stub_provider

    # Create manager
    provider = create_stub_provider(simulate_latency_ms=20)
    manager = VisionManager(vision_provider=provider, ocr_manager=None)

    # Create request
    request = VisionAnalyzeRequest(
        image_base64=sample_image_base64, include_description=True, include_ocr=False
    )

    # Analyze
    response = await manager.analyze(request)

    # Assertions
    assert response.success is True
    assert response.description is not None
    assert response.ocr is None  # No OCR manager
    assert response.provider == "deepseek_stub"
    assert response.processing_time_ms > 0
    assert response.error is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
