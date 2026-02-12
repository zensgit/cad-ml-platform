from fastapi.testclient import TestClient

from src.main import app
from src.inference import classifier_api


def test_health_classifier_cache_stats(monkeypatch):
    class DummyCache:
        def stats(self):
            return {"size": 2, "max_size": 1000, "hits": 3, "misses": 1}

    monkeypatch.setattr(classifier_api, "result_cache", DummyCache())

    client = TestClient(app)
    response = client.get("/api/v1/health/classifier/cache", headers={"X-Admin-Token": "test"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["size"] == 2
    assert payload["max_size"] == 1000
    assert payload["hits"] == 3
    assert payload["misses"] == 1
    assert payload["hit_ratio"] == 0.75
