import time
import pytest
import statistics
import numpy as np
from src.core.similarity import InMemoryVectorStore, FaissVectorStore

@pytest.mark.asyncio
async def test_vector_search_latency():
    """
    Performance test for vector search.
    Goal: < 10ms for 10k vectors.
    """
    # Setup
    dim = 64 # Metric learning dim
    count = 10000
    
    # Create synthetic data
    vectors = np.random.rand(count, dim).astype(np.float32)
    ids = [f"vec_{i}" for i in range(count)]
    
    # Test both backends if available
    stores = [InMemoryVectorStore()]
    try:
        faiss_store = FaissVectorStore()
        if faiss_store._available:
            stores.append(faiss_store)
        else:
            print("FaissVectorStore not available")
    except Exception as e:
        print(f"FaissVectorStore init failed: {e}")
        
    for store in stores:
        backend_name = store.__class__.__name__
        print(f"\nTesting {backend_name} with {count} vectors...")
        
        # Add vectors
        start_add = time.perf_counter()
        for i in range(count):
            store.add(ids[i], vectors[i].tolist())
        add_time = (time.perf_counter() - start_add) * 1000
        print(f"  Add time: {add_time:.2f} ms")
        
        # Query
        query_vec = np.random.rand(dim).astype(np.float32).tolist()
        
        latencies = []
        iterations = 100
        
        for _ in range(iterations):
            start_q = time.perf_counter()
            store.query(query_vec, top_k=10)
            end_q = time.perf_counter()
            latencies.append((end_q - start_q) * 1000)
            
        avg_latency = statistics.mean(latencies)
        p99_latency = statistics.quantiles(latencies, n=100)[98]
        
        print(f"  Query Latency (N={iterations}):")
        print(f"    Average: {avg_latency:.4f} ms")
        print(f"    P99:     {p99_latency:.4f} ms")
        
        # Thresholds
        if "Faiss" in backend_name:
            assert avg_latency < 10, f"Faiss average latency {avg_latency}ms > 10ms"
