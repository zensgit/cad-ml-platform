import pytest
import os
import psutil
import gc
import asyncio
from src.core.feature_extractor import FeatureExtractor
from src.adapters.factory import AdapterFactory
from src.models.cad_document import CadDocument

SAMPLE_FILE = "examples/sample_part.step"

@pytest.mark.asyncio
async def test_memory_leak():
    """
    Simple memory leak detection by running extraction in a loop.
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

    process = psutil.Process(os.getpid())
    extractor = FeatureExtractor()
    
    # Warmup
    await extractor.extract(doc)
    gc.collect()
    
    initial_mem = process.memory_info().rss / 1024 / 1024 # MB
    print(f"\nInitial Memory: {initial_mem:.2f} MB")
    
    iterations = 50
    for i in range(iterations):
        await extractor.extract(doc)
        if i % 10 == 0:
            gc.collect()
            
    gc.collect()
    final_mem = process.memory_info().rss / 1024 / 1024 # MB
    print(f"Final Memory:   {final_mem:.2f} MB")
    
    growth = final_mem - initial_mem
    print(f"Memory Growth:  {growth:.2f} MB")
    
    # Allow some small fluctuation, but significant growth indicates leak
    assert growth < 10.0, f"Memory grew by {growth:.2f} MB, possible leak"
