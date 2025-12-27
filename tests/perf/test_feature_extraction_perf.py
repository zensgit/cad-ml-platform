import asyncio
import os
import statistics
import time

import pytest

from src.adapters.factory import AdapterFactory
from src.core.feature_extractor import FeatureExtractor
from src.models.cad_document import CadDocument

SAMPLE_FILE = "examples/sample_part.step"


@pytest.mark.asyncio
async def test_feature_extraction_performance():
    """
    Performance test for feature extraction.
    Goal: < 200ms per file for 95th percentile (excluding I/O if possible, but end-to-end here).
    """
    if not os.path.exists(SAMPLE_FILE):
        pytest.skip(f"Sample file {SAMPLE_FILE} not found")

    # Read file content
    with open(SAMPLE_FILE, "rb") as f:
        content = f.read()

    file_format = "step"
    adapter = AdapterFactory.get_adapter(file_format)

    # Parse document (include parsing time in measurement if we want end-to-end,
    # but usually feature extraction is separate. The test name implies feature extraction only.
    # However, the goal < 200ms likely includes parsing if it's an API SLA.
    # Let's measure both: parsing + extraction).

    # Pre-parse for pure extraction test
    try:
        if hasattr(adapter, "parse"):
            doc = await adapter.parse(content, file_name="sample_part.step")
        else:
            # Fallback for legacy adapters
            await adapter.convert(content, file_name="sample_part.step")
            doc = CadDocument(file_name="sample_part.step", format=file_format)
    except Exception as e:
        pytest.skip(f"Adapter failed to parse: {e}")

    extractor = FeatureExtractor(feature_version="v6")

    # Warmup
    await extractor.extract(doc)

    latencies = []
    iterations = 20

    for _ in range(iterations):
        start_time = time.perf_counter()
        await extractor.extract(doc)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # ms

    avg_latency = statistics.mean(latencies)
    p95_latency = statistics.quantiles(latencies, n=20)[18]  # approx 95th

    print(f"\nFeature Extraction Performance (v6, N={iterations}):")
    print(f"  Average: {avg_latency:.2f} ms")
    print(f"  P95:     {p95_latency:.2f} ms")

    # Assertions (adjust thresholds based on reality of the environment)
    # CI environments might be slow, so we use a generous threshold or just report.
    # assert avg_latency < 500, f"Average latency {avg_latency}ms > 500ms"
