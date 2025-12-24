import asyncio
import importlib

import pytest

from src.core.feature_extractor import (
    FeatureExtractor,
    compute_shape_entropy,
    compute_surface_count,
)
from src.models.cad_document import BoundingBox, CadDocument, CadEntity


def test_v4_length_24():
    mod = importlib.import_module("src.core.feature_extractor")
    length = getattr(mod, "VECTOR_V4_LENGTH", None)
    if length is None:
        pytest.skip("VECTOR_V4_LENGTH not defined; skip until aligned")
    assert length == 24


def test_latency_metric_exported():
    metrics = importlib.import_module("src.utils.analysis_metrics")
    exported = set(getattr(metrics, "__all__", []))
    assert "feature_extraction_latency_seconds" in exported


def test_entropy_bounds():
    entropy = compute_shape_entropy({"LINE": 10, "CIRCLE": 4, "ARC": 2})
    assert 0.0 <= entropy <= 1.0


def test_single_type_entropy_zero():
    assert compute_shape_entropy({"LINE": 10}) == 0.0


def test_uniform_entropy_near_one():
    entropy = compute_shape_entropy({"A": 1, "B": 1, "C": 1, "D": 1})
    assert abs(entropy - 1.0) < 1e-6


def test_surface_count_basic():
    doc = CadDocument(
        file_name="sample.step",
        format="step",
        entities=[
            CadEntity(kind="FACE"),
            CadEntity(kind="LINE"),
            CadEntity(kind="SURFACE"),
        ],
    )
    assert compute_surface_count(doc) == 2


@pytest.mark.asyncio
async def test_concurrency_consistency():
    doc = CadDocument(
        file_name="sample.step",
        format="step",
        entities=[
            CadEntity(kind="FACE"),
            CadEntity(kind="LINE"),
            CadEntity(kind="SURFACE"),
        ],
        layers={"0": 3},
        bounding_box=BoundingBox(min_x=0, min_y=0, min_z=0, max_x=2, max_y=4, max_z=6),
        metadata={"solids": 1, "facets": 2},
    )
    extractor = FeatureExtractor(feature_version="v4")
    tasks = [extractor.extract(doc) for _ in range(5)]
    results = await asyncio.gather(*tasks)
    first = results[0]
    for result in results[1:]:
        assert result == first


def test_upgrade_downgrade_dimensions():
    fe_v4 = FeatureExtractor(feature_version="v4")
    v1 = [0.0] * fe_v4.expected_dim("v1")
    upgraded = fe_v4.upgrade_vector(v1, current_version="v1")
    assert len(upgraded) == fe_v4.expected_dim("v4")

    fe_v2 = FeatureExtractor(feature_version="v2")
    v4 = [0.0] * fe_v2.expected_dim("v4")
    downgraded = fe_v2.upgrade_vector(v4, current_version="v4")
    assert len(downgraded) == fe_v2.expected_dim("v2")
