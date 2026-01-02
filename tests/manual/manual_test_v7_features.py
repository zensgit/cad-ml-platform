import asyncio
import logging
import os
import sys

# Add src to path
sys.path.append(os.getcwd())

from src.core.feature_extractor import FeatureExtractor
from src.models.cad_document import BoundingBox, CadDocument, CadEntity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_test():
    logger.info("Testing v7 Feature Extraction...")

    # 1. Setup Environment
    os.environ["FEATURE_VERSION"] = "v7"
    os.environ["RENDERER_BACKEND"] = "dummy"  # Force dummy for stability

    # 2. Create Dummy Document
    doc = CadDocument(
        file_name="test.dxf",
        format="dxf",
        entities=[
            CadEntity(kind="LINE", attributes={"start": [0, 0, 0], "end": [10, 0, 0]}),
            CadEntity(kind="CIRCLE", attributes={"center": [5, 5, 0], "radius": 2.0}),
        ],
        layers={"0": 2},
        metadata={"solids": 1, "facets": 10},
    )
    doc.bounding_box = BoundingBox(min_x=0, min_y=0, min_z=0, max_x=10, max_y=10, max_z=1)

    # 3. Extract
    extractor = FeatureExtractor(feature_version="v7")
    features = await extractor.extract(doc)

    # 4. Validate
    geo = features["geometric"]
    sem = features["semantic"]

    logger.info(f"Geometric Dim: {len(geo)}")
    logger.info(f"Semantic Dim: {len(sem)}")

    # v7 = 160 total dims (158 geometric + 2 semantic)
    assert len(geo) == 158, f"Expected 158 geometric dims, got {len(geo)}"
    assert len(sem) == 2, f"Expected 2 semantic dims, got {len(sem)}"

    # Check visual part (last 128 dims)
    # Dummy renderer returns 0.0s
    visual_part = geo[-128:]
    assert all(x == 0.0 for x in visual_part), "Expected zero vector from DummyRenderer"

    logger.info("✅ v7 Extraction Successful (Dummy Renderer)")


if __name__ == "__main__":
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_test())
        loop.close()
    except Exception as e:
        logger.error(f"Test Failed: {e}")
        sys.exit(1)
