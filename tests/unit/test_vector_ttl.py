import os
import time
import uuid
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.main import app
from src.core import similarity
from src.core.similarity import register_vector, InMemoryVectorStore

client = TestClient(app)


def test_vector_ttl_pruning():
    # _TTL_SECONDS is set at module load time, so we need to patch the module variable
    vid = str(uuid.uuid4())

    # Patch the module-level _TTL_SECONDS to enable TTL-based pruning
    with patch.object(similarity, "_TTL_SECONDS", 1):
        register_vector(vid, [0.1, 0.2, 0.3], meta={"material": "steel"})
        # Also need to add timestamp
        similarity._VECTOR_TS[vid] = time.time() - 2  # Set timestamp to 2 seconds ago

        store = InMemoryVectorStore()
        assert store.exists(vid)

        # trigger prune via query
        store.query([0.1, 0.2, 0.3], top_k=1)
        assert not store.exists(vid)
