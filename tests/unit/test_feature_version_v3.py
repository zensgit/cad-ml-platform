import asyncio
import os

from src.core.feature_extractor import FeatureExtractor
from src.models.cad_document import BoundingBox, CadDocument, CadEntity


def test_feature_version_v3_dimensions():
    os.environ["FEATURE_VERSION"] = "v3"
    doc = CadDocument(file_name="sample.step", format="step")
    # simulate geometry metadata
    doc.metadata["solids"] = 3
    doc.metadata["facets"] = 10
    # add entities of different kinds
    doc.entities.extend(
        [
            CadEntity(kind="SOLID"),
            CadEntity(kind="SOLID"),
            CadEntity(kind="SOLID"),
            CadEntity(kind="FACET"),
            CadEntity(kind="FACET"),
            CadEntity(kind="LINE"),
        ]
    )
    doc.bounding_box.max_x = 10
    doc.bounding_box.max_y = 5
    doc.bounding_box.max_z = 2
    extractor = FeatureExtractor()
    features = asyncio.run(extractor.extract(doc))
    geom = features["geometric"]
    # Base v1: 5 values + v2 adds 5 + v3 adds solids/facets/3 stats + top5 kind frequencies (5)
    # total expected: 5 + 5 + (5 enrichment) + 5 freq = 20
    assert len(geom) == 20, f"Unexpected dimension {len(geom)}: {geom}"
