import asyncio

from src.core.feature_extractor import FeatureExtractor
from src.models.cad_document import BoundingBox, CadDocument, CadEntity


def test_feature_extractor_basic():
    doc = CadDocument(
        file_name="sample.dxf",
        format="dxf",
        entities=[CadEntity(kind="LINE"), CadEntity(kind="CIRCLE")],
        layers={"0": 2},
        bounding_box=BoundingBox(min_x=0, min_y=0, min_z=0, max_x=10, max_y=5, max_z=2),
    )
    extractor = FeatureExtractor()
    features = asyncio.run(extractor.extract(doc))
    assert features["geometric"][0] == 2  # entity count
    assert features["geometric"][4] == 10 * 5 * 2  # volume estimate
    assert features["semantic"][0] == 1  # layer count
