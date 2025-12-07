import pytest
import asyncio
import time
import statistics
import os
from src.core.feature_extractor import FeatureExtractor
from src.adapters.factory import AdapterFactory
from src.models.cad_document import CadDocument

SAMPLE_FILE = "examples/sample_part.step"

@pytest.mark.asyncio
async def test_concurrency_stress():
    """
    Stress test with concurrent extraction requests.
    """
    if not os.path.exists(SAMPLE_FILE):
        pytest.skip(f"Sample file {SAMPLE_FILE} not found")

    # Read file content
    with open(SAMPLE_FILE, "rb") as f:
        content = f.read()
    
    file_format = "step"
    adapter = AdapterFactory.get_adapter(file_format)
    
    # Pre-parse
    try:
        if hasattr(adapter, "parse"):
            doc = await adapter.parse(content, file_name="sample_part.step")
        else:
            await adapter.convert(content, file_name="sample_part.step")
            doc = CadDocument(file_name="sample_part.step", format=file_format)
    except Exception as e:
        pytest.skip(f"Adapter failed to parse: {e}")

    extractor = FeatureExtractor()
    concurrency = 10
    total_requests = 50
    
    async def worker():
        start = time.perf_counter()
        await extractor.extract(doc)
        return time.perf_counter() - start

    print(f"\nStarting concurrency test (C={concurrency}, N={total_requests})...")
    
    start_total = time.perf_counter()
    
    # To simulate limited concurrency, we can use a semaphore
    sem = asyncio.Semaphore(concurrency)
    
    async def bounded_worker():
        async with sem:
            return await worker()
            
    tasks = [bounded_worker() for _ in range(total_requests)]
    results = await asyncio.gather(*tasks)
    
    total_time = time.perf_counter() - start_total
    
    latencies_ms = [r * 1000 for r in results]
    avg_latency = statistics.mean(latencies_ms)
    p95_latency = statistics.quantiles(latencies_ms, n=20)[18]
    throughput = total_requests / total_time
    
    print(f"Total Time: {total_time:.2f} s")
    print(f"Throughput: {throughput:.2f} req/s")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    print(f"P95 Latency: {p95_latency:.2f} ms")
    
    assert throughput > 1.0, "Throughput too low"
