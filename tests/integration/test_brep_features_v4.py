"""Integration coverage for L3 B-Rep surface metrics in v4 extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

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


def _assert_bbox_dims(brep_features: Dict[str, Any], expected: Tuple[float, float, float]) -> None:
    bbox = brep_features["bbox"]
    assert bbox["x"] == pytest.approx(expected[0], rel=1e-6, abs=1e-5)
    assert bbox["y"] == pytest.approx(expected[1], rel=1e-6, abs=1e-5)
    assert bbox["z"] == pytest.approx(expected[2], rel=1e-6, abs=1e-5)


def _assert_surface_types(brep_features: Dict[str, Any], expected: Dict[str, int]) -> None:
    surface_types = brep_features["surface_types"]
    for key, count in expected.items():
        assert surface_types.get(key) == count
    assert sum(surface_types.values()) == brep_features["faces"]


async def _extract_surface_metrics(step_path: Path) -> Tuple[Dict[str, Any], float, float]:
    engine = get_geometry_engine()
    shape = engine.load_step(step_path.read_bytes(), file_name=step_path.name)
    assert shape is not None

    brep_features = engine.extract_brep_features(shape)
    assert brep_features.get("valid_3d") is True
    assert brep_features.get("faces", 0) > 0

    doc = CadDocument(file_name=step_path.name, format="step")
    extractor = FeatureExtractor(feature_version="v4")
    features = await extractor.extract(doc, brep_features=brep_features)
    geometric = features["geometric"]
    return brep_features, float(geometric[-2]), float(geometric[-1])


@pytest.mark.asyncio
async def test_brep_surface_metrics_from_generated_steps(tmp_path: Path) -> None:
    from OCC.Core.BRepPrimAPI import (
        BRepPrimAPI_MakeBox,
        BRepPrimAPI_MakeCylinder,
        BRepPrimAPI_MakeSphere,
        BRepPrimAPI_MakeTorus,
    )

    box_path = tmp_path / "box.step"
    cylinder_path = tmp_path / "cylinder.step"
    sphere_path = tmp_path / "sphere.step"
    torus_path = tmp_path / "torus.step"

    _write_step(BRepPrimAPI_MakeBox(10, 20, 30).Shape(), box_path)
    _write_step(BRepPrimAPI_MakeCylinder(5, 20).Shape(), cylinder_path)
    _write_step(BRepPrimAPI_MakeSphere(7).Shape(), sphere_path)
    _write_step(BRepPrimAPI_MakeTorus(10, 3).Shape(), torus_path)

    box_brep, box_surface_count, box_entropy = await _extract_surface_metrics(box_path)
    assert int(box_surface_count) == box_brep["faces"]
    _assert_bbox_dims(box_brep, (10.0, 20.0, 30.0))
    _assert_surface_types(box_brep, {"plane": 6})
    assert box_entropy == pytest.approx(0.0)

    cylinder_brep, cylinder_surface_count, cylinder_entropy = await _extract_surface_metrics(
        cylinder_path
    )
    assert int(cylinder_surface_count) == cylinder_brep["faces"]
    _assert_bbox_dims(cylinder_brep, (10.0, 10.0, 20.0))
    _assert_surface_types(cylinder_brep, {"plane": 2, "cylinder": 1})
    assert 0.0 < cylinder_entropy <= 1.0

    sphere_brep, sphere_surface_count, sphere_entropy = await _extract_surface_metrics(
        sphere_path
    )
    assert int(sphere_surface_count) == sphere_brep["faces"]
    _assert_bbox_dims(sphere_brep, (14.0, 14.0, 14.0))
    _assert_surface_types(sphere_brep, {"sphere": sphere_brep["faces"]})
    assert sphere_entropy == pytest.approx(0.0)

    torus_brep, torus_surface_count, torus_entropy = await _extract_surface_metrics(
        torus_path
    )
    assert int(torus_surface_count) == torus_brep["faces"]
    _assert_bbox_dims(torus_brep, (26.0, 26.0, 6.0))
    _assert_surface_types(torus_brep, {"torus": torus_brep["faces"]})
    assert torus_entropy == pytest.approx(0.0)
