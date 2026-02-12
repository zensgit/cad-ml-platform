from fastapi.testclient import TestClient

from src.inference import classifier_api


def test_classifier_cache_hits(monkeypatch):
    call_count = {"n": 0}

    def fake_load():
        return None

    def fake_predict(_path: str):
        call_count["n"] += 1
        return {
            "category": "其他",
            "confidence": 0.9,
            "probabilities": {"其他": 0.9},
        }

    monkeypatch.setattr(classifier_api.classifier, "load", fake_load)
    monkeypatch.setattr(classifier_api.classifier, "predict", fake_predict)

    client = TestClient(classifier_api.app)

    admin_headers = {"X-Admin-Token": "test"}
    response = client.post("/cache/clear", headers=admin_headers)
    assert response.status_code == 200

    payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"

    response = client.post(
        "/classify",
        files={"file": ("sample.dxf", payload, "application/dxf")},
    )
    assert response.status_code == 200

    response = client.post(
        "/classify",
        files={"file": ("sample.dxf", payload, "application/dxf")},
    )
    assert response.status_code == 200

    assert call_count["n"] == 1

    stats = client.get("/cache/stats", headers=admin_headers)
    assert stats.status_code == 200
    data = stats.json()
    assert data["hits"] >= 1
    assert data["misses"] >= 1


def test_classifier_rate_limit(monkeypatch):
    class DummyLimiter:
        def __init__(self):
            self.calls = 0

        def allow(self, _key: str) -> bool:
            self.calls += 1
            return self.calls <= 1

    monkeypatch.setattr(classifier_api, "_rate_limiter", DummyLimiter())

    client = TestClient(classifier_api.app)
    payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"

    response = client.post(
        "/classify",
        files={"file": ("sample.dxf", payload, "application/dxf")},
    )
    assert response.status_code in (200, 400, 500)

    response = client.post(
        "/classify",
        files={"file": ("sample.dxf", payload, "application/dxf")},
    )
    assert response.status_code == 429


class TestLRUCacheUnit:
    """Unit tests for LRUCache class."""

    def test_cache_init(self):
        """Test cache initialization."""
        cache = classifier_api.LRUCache(max_size=100)
        assert cache.max_size == 100
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        cache = classifier_api.LRUCache(max_size=10)
        content = b"test content"
        result = {"category": "壳体类", "confidence": 0.95}

        cache.put(content, result)
        assert len(cache.cache) == 1

        cached = cache.get(content)
        assert cached == result
        assert cache.hits == 1
        assert cache.misses == 0

    def test_cache_miss(self):
        """Test cache miss."""
        cache = classifier_api.LRUCache(max_size=10)

        cached = cache.get(b"unknown content")
        assert cached is None
        assert cache.misses == 1
        assert cache.hits == 0

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = classifier_api.LRUCache(max_size=3)

        for i in range(3):
            cache.put(f"content_{i}".encode(), {"id": i})

        assert len(cache.cache) == 3

        cache.put(b"content_new", {"id": "new"})
        assert len(cache.cache) == 3

        # First item should be evicted
        assert cache.get(b"content_0") is None
        assert cache.get(b"content_new") is not None

    def test_cache_move_to_end_on_access(self):
        """Test that accessed items move to end (LRU behavior)."""
        cache = classifier_api.LRUCache(max_size=3)

        cache.put(b"a", {"v": "a"})
        cache.put(b"b", {"v": "b"})
        cache.put(b"c", {"v": "c"})

        # Access 'a' - should move to end
        cache.get(b"a")

        # Add new item - should evict 'b' (now oldest)
        cache.put(b"d", {"v": "d"})

        assert cache.get(b"b") is None  # evicted
        assert cache.get(b"a") is not None  # still present

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = classifier_api.LRUCache(max_size=100)

        cache.put(b"content", {"result": 1})
        cache.get(b"content")  # hit
        cache.get(b"content")  # hit
        cache.get(b"unknown")  # miss

        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 100
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert "66" in stats["hit_rate"]

    def test_cache_clear(self):
        """Test cache clear."""
        cache = classifier_api.LRUCache(max_size=10)
        cache.put(b"a", {"v": 1})
        cache.put(b"b", {"v": 2})
        cache.get(b"a")
        cache.get(b"unknown")

        cache.clear()

        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_update_existing(self):
        """Test updating existing cache entry."""
        cache = classifier_api.LRUCache(max_size=10)
        content = b"content"

        cache.put(content, {"v": 1})
        cache.put(content, {"v": 2})

        cached = cache.get(content)
        assert cached["v"] == 2
        assert len(cache.cache) == 1


def test_batch_classify_uses_cache(monkeypatch):
    """Test batch endpoint uses cache."""
    call_count = {"n": 0}

    def fake_load():
        return None

    def fake_predict(_path: str):
        call_count["n"] += 1
        return {
            "category": "轴类",
            "confidence": 0.85,
            "probabilities": {"轴类": 0.85},
        }

    monkeypatch.setattr(classifier_api.classifier, "load", fake_load)
    monkeypatch.setattr(classifier_api.classifier, "predict", fake_predict)

    client = TestClient(classifier_api.app)

    admin_headers = {"X-Admin-Token": "test"}
    client.post("/cache/clear", headers=admin_headers)

    payload1 = b"0\nSECTION\n2\nENTITIES\n0\nLINE\n0\nENDSEC\n0\nEOF\n"
    payload2 = b"0\nSECTION\n2\nENTITIES\n0\nCIRCLE\n0\nENDSEC\n0\nEOF\n"

    # First batch - 2 different files
    response = client.post(
        "/classify/batch",
        files=[
            ("files", ("file1.dxf", payload1, "application/dxf")),
            ("files", ("file2.dxf", payload2, "application/dxf")),
        ],
    )
    assert response.status_code == 200
    assert call_count["n"] == 2

    # Second batch - same files (should hit cache)
    response = client.post(
        "/classify/batch",
        files=[
            ("files", ("file1.dxf", payload1, "application/dxf")),
            ("files", ("file2.dxf", payload2, "application/dxf")),
        ],
    )
    assert response.status_code == 200
    # Should still be 2 (no new predictions)
    assert call_count["n"] == 2


def test_root_endpoint(monkeypatch):
    """Test health check endpoint."""
    def fake_load():
        return None

    monkeypatch.setattr(classifier_api.classifier, "load", fake_load)
    client = TestClient(classifier_api.app)

    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "accuracy" in data


def test_categories_endpoint(monkeypatch):
    """Test categories endpoint."""
    def fake_load():
        return None

    monkeypatch.setattr(classifier_api.classifier, "load", fake_load)
    client = TestClient(classifier_api.app)

    response = client.get("/categories")
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert len(data["categories"]) == 5
    assert "传动件" in data["categories"]
    assert "壳体类" in data["categories"]


def test_classify_non_dxf_file(monkeypatch):
    """Test classify rejects non-DXF files."""
    def fake_load():
        return None

    monkeypatch.setattr(classifier_api.classifier, "load", fake_load)
    client = TestClient(classifier_api.app)

    response = client.post(
        "/classify",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "DXF" in response.json()["detail"]


def test_batch_classify_mixed_files(monkeypatch):
    """Test batch classify handles mixed file types."""
    call_count = {"n": 0}

    def fake_load():
        return None

    def fake_predict(_path: str):
        call_count["n"] += 1
        return {
            "category": "连接件",
            "confidence": 0.88,
            "probabilities": {"连接件": 0.88},
        }

    monkeypatch.setattr(classifier_api.classifier, "load", fake_load)
    monkeypatch.setattr(classifier_api.classifier, "predict", fake_predict)

    client = TestClient(classifier_api.app)
    admin_headers = {"X-Admin-Token": "test"}
    client.post("/cache/clear", headers=admin_headers)

    payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"

    response = client.post(
        "/classify/batch",
        files=[
            ("files", ("valid.dxf", payload, "application/dxf")),
            ("files", ("invalid.txt", b"hello", "text/plain")),
        ],
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert data["success"] == 1  # Only DXF file succeeds
    # Check that non-DXF file has error
    for r in data["results"]:
        if r["filename"] == "invalid.txt":
            assert r["category"] == "error"


def test_batch_classify_duplicate_filenames(monkeypatch):
    """Ensure batch results preserve order even with duplicate filenames."""
    def fake_load():
        return None

    def fake_predict(path: str):
        with open(path, "rb") as handle:
            content = handle.read()
        if b"DXF-A" in content:
            return {
                "category": "壳体类",
                "confidence": 0.91,
                "probabilities": {"壳体类": 0.91},
            }
        return {
            "category": "轴类",
            "confidence": 0.88,
            "probabilities": {"轴类": 0.88},
        }

    monkeypatch.setattr(classifier_api.classifier, "load", fake_load)
    monkeypatch.setattr(classifier_api.classifier, "predict", fake_predict)

    client = TestClient(classifier_api.app)
    admin_headers = {"X-Admin-Token": "test"}
    client.post("/cache/clear", headers=admin_headers)

    payload_a = b"0\nSECTION\n2\nENTITIES\n0\nDXF-A\n0\nENDSEC\n0\nEOF\n"
    payload_b = b"0\nSECTION\n2\nENTITIES\n0\nDXF-B\n0\nENDSEC\n0\nEOF\n"

    response = client.post(
        "/classify/batch",
        files=[
            ("files", ("dup.dxf", payload_a, "application/dxf")),
            ("files", ("dup.dxf", payload_b, "application/dxf")),
        ],
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["category"] == "壳体类"
    assert data["results"][1]["category"] == "轴类"


def test_warmup_model_function():
    """Test that warmup function exists and is callable."""
    assert hasattr(classifier_api, "_warmup_model")
    assert callable(classifier_api._warmup_model)


def test_executor_exists():
    """Test that thread pool executor is configured."""
    assert hasattr(classifier_api, "_executor")
    assert classifier_api._executor is not None


class TestHybridCache:
    """Tests for HybridCache class."""

    def test_hybrid_cache_init(self):
        """Test hybrid cache initialization."""
        cache = classifier_api.HybridCache(l1_max_size=100)
        assert cache.l1.max_size == 100
        assert hasattr(cache, "_redis_available")

    def test_hybrid_cache_l1_only(self):
        """Test hybrid cache works with L1 only (no Redis)."""
        cache = classifier_api.HybridCache(l1_max_size=10)
        content = b"test content for hybrid"
        result = {"category": "传动件", "confidence": 0.91}

        # Put and get
        cache.put(content, result)
        cached = cache.get(content)

        assert cached == result
        assert cache.l1.hits >= 1

    def test_hybrid_cache_stats(self):
        """Test hybrid cache stats include redis info."""
        cache = classifier_api.HybridCache(l1_max_size=10)
        stats = cache.stats()

        assert "size" in stats
        assert "redis_enabled" in stats

    def test_hybrid_cache_clear(self):
        """Test hybrid cache clear."""
        cache = classifier_api.HybridCache(l1_max_size=10)
        cache.put(b"content1", {"v": 1})
        cache.put(b"content2", {"v": 2})

        cache.clear()

        assert len(cache.l1.cache) == 0

    def test_result_cache_is_hybrid(self):
        """Test global result_cache is HybridCache instance."""
        assert isinstance(classifier_api.result_cache, classifier_api.HybridCache)
