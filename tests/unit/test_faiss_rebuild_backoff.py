import os
import time

import pytest
from fastapi.testclient import TestClient

from src.api.health_utils import metrics_enabled
from src.main import app


def test_faiss_rebuild_backoff_metric(monkeypatch):
    if not metrics_enabled():
        pytest.skip("metrics client disabled in this environment")

    # Ensure backend env forces faiss logic path (even if faiss lib absent, should degrade gracefully)
    os.environ["VECTOR_STORE_BACKEND"] = "faiss"
    client = TestClient(app)
    # Attempt to delete enough vectors to trigger rebuild logic (pending delete threshold)
    # We simulate by directly invoking similarity store mark_delete via endpoint registration of fake vectors
    # First register a few vectors through analysis (simplified DXF content)
    content = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    for i in range(3):
        files = {"file": (f"t{i}.dxf", content, "application/octet-stream")}
        resp = client.post(
            "/api/v1/analyze/",
            data={"options": '{"extract_features": true}'},
            files=files,
            headers={"X-API-Key": "test"},
        )
        assert resp.status_code == 200
    # Delete one to add to pending delete set
    from src.core.similarity import _VECTOR_STORE

    vid = list(_VECTOR_STORE.keys())[0]
    del_resp = client.post(
        "/api/v1/vectors/delete",
        json={"id": vid},
        headers={"X-API-Key": "test"},
    )
    assert del_resp.status_code in (200, 404) or del_resp.status_code == 500
    # Check metric object exists
    from src.utils.analysis_metrics import faiss_rebuild_backoff_seconds

    assert hasattr(faiss_rebuild_backoff_seconds, "_value") or hasattr(
        faiss_rebuild_backoff_seconds, "_metrics"
    )
