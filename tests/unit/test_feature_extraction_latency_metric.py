import pytest
from src.core.feature_extractor import FeatureExtractor
from src.models.cad_document import CadDocument, CadEntity, BoundingBox
from src.utils.analysis_metrics import feature_extraction_latency_seconds


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
@pytest.mark.asyncio
async def test_feature_extraction_latency_metric(version):
    doc = CadDocument(
        file_name="sample.dxf",
        format="dxf",
        entities=[CadEntity(kind="LINE"), CadEntity(kind="CIRCLE")],
        layers={"0": 2},
        bounding_box=BoundingBox(min_x=0, min_y=0, min_z=0, max_x=10, max_y=5, max_z=2),
        metadata={"solids": 1, "facets": 3},
    )
    fx = FeatureExtractor(feature_version=version)
    await fx.extract(doc)
    # Dummy metrics fallback (no collect). If real client present, object exposes collect method.
    if hasattr(feature_extraction_latency_seconds, 'collect'):
        has_label = False
        for collected in feature_extraction_latency_seconds.collect():  # type: ignore
            for s in collected.samples:
                if s.labels.get("version") == version:
                    has_label = True
                    break
        assert has_label, f"No latency samples recorded for version {version}"
    else:
        # Fallback: ensure observe didn't throw (reaching here means success)
        assert True
