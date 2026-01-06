from __future__ import annotations

from PIL import Image, ImageDraw
import pytest

from src.core.vision_analyzer import VisionAnalyzer


@pytest.mark.asyncio
async def test_extract_cad_features_detects_line_and_circle() -> None:
    image = Image.new("L", (120, 80), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((10, 10, 110, 10), fill=0, width=3)
    draw.ellipse((60, 40, 80, 60), fill=0)

    analyzer = VisionAnalyzer()
    features = await analyzer._extract_cad_features(image)

    assert features["dimensions"]["overall_width"] == 120
    assert features["dimensions"]["overall_height"] == 80
    assert len(features["drawings"]["lines"]) >= 1
    assert len(features["drawings"]["circles"]) >= 1
    assert features["stats"]["components"] == (
        len(features["drawings"]["lines"]) + len(features["drawings"]["circles"])
    )


@pytest.mark.asyncio
async def test_extract_cad_features_handles_blank_image() -> None:
    image = Image.new("L", (64, 64), color=255)

    analyzer = VisionAnalyzer()
    features = await analyzer._extract_cad_features(image)

    assert features["drawings"]["lines"] == []
    assert features["drawings"]["circles"] == []
    assert features["stats"]["ink_ratio"] == 0.0
    assert features["stats"]["components"] == 0
