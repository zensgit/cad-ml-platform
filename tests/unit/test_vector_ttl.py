import os
import time
import uuid
from fastapi.testclient import TestClient

from src.main import app
from src.core.similarity import register_vector, InMemoryVectorStore

client = TestClient(app)


def test_vector_ttl_pruning():
    os.environ["VECTOR_TTL_SECONDS"] = "1"  # expire quickly
    vid = str(uuid.uuid4())
    register_vector(vid, [0.1, 0.2, 0.3], meta={"material": "steel"})
    store = InMemoryVectorStore()
    assert store.exists(vid)
    time.sleep(1.2)
    # trigger prune via query
    store.query([0.1, 0.2, 0.3], top_k=1)
    assert not store.exists(vid)
    os.environ.pop("VECTOR_TTL_SECONDS", None)
