from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from src.core.vision_analyzer import VisionAnalyzer

FIXTURES_DIR = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "vision"
    / "fixtures"
    / "cad_features"
)


async def _extract_stats(image_name: str) -> dict:
    analyzer = VisionAnalyzer(initialize_clients=False)
    with Image.open(FIXTURES_DIR / image_name) as image:
        features = await analyzer._extract_cad_features(image)
    return analyzer._summarize_cad_features(features)


@pytest.mark.asyncio
async def test_cad_feature_golden_line_stats() -> None:
    stats = await _extract_stats("cad_line.png")

    assert stats["line_count"] == 1
    assert stats["circle_count"] == 0
    assert stats["arc_count"] == 0
    assert stats["line_angle_bins"]["0-30"] == 1
    assert sum(stats["line_angle_bins"].values()) == 1
    assert stats["line_angle_avg"] == 0.0
    assert sum(stats["arc_sweep_bins"].values()) == 0
    assert stats["arc_sweep_avg"] is None


@pytest.mark.asyncio
async def test_cad_feature_golden_circle_stats() -> None:
    stats = await _extract_stats("cad_circle.png")

    assert stats["line_count"] == 0
    assert stats["circle_count"] == 1
    assert stats["arc_count"] == 0
    assert sum(stats["line_angle_bins"].values()) == 0
    assert stats["line_angle_avg"] is None
    assert sum(stats["arc_sweep_bins"].values()) == 0
    assert stats["arc_sweep_avg"] is None


@pytest.mark.asyncio
async def test_cad_feature_golden_arc_stats() -> None:
    stats = await _extract_stats("cad_arc.png")

    assert stats["line_count"] == 0
    assert stats["circle_count"] == 0
    assert stats["arc_count"] == 1
    assert sum(stats["line_angle_bins"].values()) == 0
    assert stats["line_angle_avg"] is None
    assert stats["arc_sweep_bins"]["180-270"] == 1
    assert sum(stats["arc_sweep_bins"].values()) == 1
    assert stats["arc_sweep_avg"] is not None
    assert 220.0 <= stats["arc_sweep_avg"] <= 260.0


@pytest.mark.asyncio
async def test_cad_feature_golden_diagonal_line_stats() -> None:
    stats = await _extract_stats("cad_line_diagonal.png")

    assert stats["line_count"] == 1
    assert stats["circle_count"] == 0
    assert stats["arc_count"] == 0
    assert stats["line_angle_bins"]["120-150"] == 1
    assert sum(stats["line_angle_bins"].values()) == 1
    assert stats["line_angle_avg"] == 135.0


@pytest.mark.asyncio
async def test_cad_feature_golden_mid_arc_stats() -> None:
    stats = await _extract_stats("cad_arc_mid.png")

    assert stats["line_count"] == 0
    assert stats["circle_count"] == 0
    assert stats["arc_count"] == 1
    assert sum(stats["line_angle_bins"].values()) == 0
    assert stats["line_angle_avg"] is None
    assert stats["arc_sweep_bins"]["90-180"] == 1
    assert sum(stats["arc_sweep_bins"].values()) == 1
    assert stats["arc_sweep_avg"] is not None
    assert 160.0 <= stats["arc_sweep_avg"] <= 180.0
