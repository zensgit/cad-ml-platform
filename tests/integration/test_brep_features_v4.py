"""Integration coverage for L3 B-Rep surface metrics in v4 extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import pytest

from src.core.feature_extractor import FeatureExtractor
from src.core.geometry.engine import HAS_OCC, get_geometry_engine
from src.models.cad_document import CadDocument

pytestmark = pytest.mark.skipif(not HAS_OCC, reason="pythonocc-core not installed")


def _write_step(shape: Any, path: Path) -> None:
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(str(path))
    assert status == IFSelect_RetDone


async def _extract_surface_metrics(step_path: Path) -> Tuple[float, float]:
    engine = get_geometry_engine()
    shape = engine.load_step(step_path.read_bytes(), file_name=step_path.name)
    assert shape is not None

    brep_features = engine.extract_brep_features(shape)
    assert brep_features.get("valid_3d") is True

    doc = CadDocument(file_name=step_path.name, format="step")
    extractor = FeatureExtractor(feature_version="v4")
    features = await extractor.extract(doc, brep_features=brep_features)
    geometric = features["geometric"]
    return float(geometric[-2]), float(geometric[-1])


@pytest.mark.asyncio
async def test_brep_surface_metrics_from_generated_steps(tmp_path: Path) -> None:
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder

    box_path = tmp_path / "box.step"
    cylinder_path = tmp_path / "cylinder.step"

    _write_step(BRepPrimAPI_MakeBox(10, 20, 30).Shape(), box_path)
    _write_step(BRepPrimAPI_MakeCylinder(5, 20).Shape(), cylinder_path)

    box_surface_count, box_entropy = await _extract_surface_metrics(box_path)
    assert box_surface_count > 0
    assert box_entropy == pytest.approx(0.0)

    cylinder_surface_count, cylinder_entropy = await _extract_surface_metrics(cylinder_path)
    assert cylinder_surface_count > 0
    assert 0.0 < cylinder_entropy <= 1.0
