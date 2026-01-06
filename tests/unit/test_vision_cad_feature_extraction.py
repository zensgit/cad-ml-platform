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
    primary_line = features["drawings"]["lines"][0]
    assert 0.0 <= primary_line["angle_degrees"] <= 10.0
    assert features["stats"]["components"] == (
        len(features["drawings"]["lines"])
        + len(features["drawings"]["circles"])
        + len(features["drawings"]["arcs"])
    )


@pytest.mark.asyncio
async def test_extract_cad_features_handles_blank_image() -> None:
    image = Image.new("L", (64, 64), color=255)

    analyzer = VisionAnalyzer()
    features = await analyzer._extract_cad_features(image)

    assert features["drawings"]["lines"] == []
    assert features["drawings"]["circles"] == []
    assert features["drawings"]["arcs"] == []
    assert features["stats"]["ink_ratio"] == 0.0
    assert features["stats"]["components"] == 0


@pytest.mark.asyncio
async def test_extract_cad_features_detects_diagonal_line() -> None:
    image = Image.new("L", (100, 100), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((10, 90, 90, 10), fill=0, width=3)

    analyzer = VisionAnalyzer()
    features = await analyzer._extract_cad_features(image)

    assert len(features["drawings"]["lines"]) >= 1
    angles = [line["angle_degrees"] for line in features["drawings"]["lines"]]
    assert any(110.0 <= angle <= 160.0 for angle in angles)


@pytest.mark.asyncio
async def test_extract_cad_features_detects_arc() -> None:
    image = Image.new("L", (100, 100), color=255)
    draw = ImageDraw.Draw(image)
    draw.arc((20, 20, 80, 80), start=0, end=180, fill=0, width=4)

    analyzer = VisionAnalyzer()
    features = await analyzer._extract_cad_features(image)

    assert len(features["drawings"]["arcs"]) >= 1
    sweeps = [arc["sweep_angle_degrees"] for arc in features["drawings"]["arcs"]]
    assert any(sweep is not None and 120.0 <= sweep <= 240.0 for sweep in sweeps)


@pytest.mark.asyncio
async def test_extract_cad_features_respects_threshold_overrides() -> None:
    image = Image.new("L", (120, 80), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((10, 10, 110, 10), fill=0, width=3)

    analyzer = VisionAnalyzer()
    features = await analyzer._extract_cad_features(
        image, {"line_aspect": 1000000.0, "line_elongation": 1000000.0}
    )

    assert features["drawings"]["lines"] == []
