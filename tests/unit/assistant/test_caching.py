"""Tests for caching module."""

import time
from typing import Any, Dict, List

import pytest

from src.core.assistant.caching import (
    CacheEntry,
    LRUCache,
    EmbeddingCache,
    SearchResultCache,
    ResponseCache,
    CacheManager,
    cached,
)


class TestCacheEntry:
    """Tests for CacheEntry class."""

    def test_creation(self):
        """Test cache entry creation."""
        entry = CacheEntry(value="test")
        assert entry.value == "test"
        assert entry.access_count == 0
        assert not entry.is_expired()

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        entry = CacheEntry(value="test", ttl=0.1)
        assert not entry.is_expired()
        time.sleep(0.15)
        assert entry.is_expired()

    def test_no_ttl_never_expires(self):
        """Test entry without TTL never expires."""
        entry = CacheEntry(value="test", ttl=None)
        assert not entry.is_expired()

    def test_touch_updates_access(self):
        """Test touch updates access metadata."""
        entry = CacheEntry(value="test")
        initial_time = entry.accessed_at
        time.sleep(0.01)
        entry.touch()

        assert entry.access_count == 1
        assert entry.accessed_at > initial_time


class TestLRUCache:
    """Tests for LRUCache class."""

    def test_basic_get_set(self):
        """Test basic get and set operations."""
        cache: LRUCache[str] = LRUCache(max_size=10)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_miss_returns_none(self):
        """Test cache miss returns None."""
        cache: LRUCache[str] = LRUCache()
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        """Test LRU eviction when at capacity."""
        cache: LRUCache[str] = LRUCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add new item, should evict key2 (least recently used)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache: LRUCache[str] = LRUCache(default_ttl=0.1)

        cache.set("key", "value")
        assert cache.get("key") == "value"

        time.sleep(0.15)
        assert cache.get("key") is None

    def test_delete(self):
        """Test delete operation."""
        cache: LRUCache[str] = LRUCache()

        cache.set("key", "value")
        assert cache.delete("key") is True
        assert cache.get("key") is None
        assert cache.delete("key") is False

    def test_clear(self):
        """Test clear operation."""
        cache: LRUCache[str] = LRUCache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert cache.size() == 0
        assert cache.get("key1") is None

    def test_stats(self):
        """Test cache statistics."""
        cache: LRUCache[str] = LRUCache(max_size=10)

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache: LRUCache[str] = LRUCache(default_ttl=0.1)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        time.sleep(0.15)

        removed = cache.cleanup_expired()
        assert removed == 2
        assert cache.size() == 0


class TestEmbeddingCache:
    """Tests for EmbeddingCache class."""

    def test_basic_operations(self):
        """Test basic cache operations."""
        cache = EmbeddingCache(max_size=100)

        embedding = [0.1, 0.2, 0.3]
        cache.set("测试文本", embedding)

        result = cache.get("测试文本")
        assert result == embedding

    def test_miss_returns_none(self):
        """Test cache miss returns None."""
        cache = EmbeddingCache()
        assert cache.get("不存在") is None

    def test_batch_operations(self):
        """Test batch get and set."""
        cache = EmbeddingCache()

        texts = ["文本1", "文本2", "文本3"]
        embeddings = [[0.1], [0.2], [0.3]]

        cache.set_batch(texts, embeddings)

        results = cache.get_batch(texts)
        assert results["文本1"] == [0.1]
        assert results["文本2"] == [0.2]
        assert results["文本3"] == [0.3]

    def test_content_hashing(self):
        """Test that same content produces same cache key."""
        cache = EmbeddingCache()

        cache.set("相同文本", [0.1])
        assert cache.get("相同文本") == [0.1]


class TestSearchResultCache:
    """Tests for SearchResultCache class."""

    def test_basic_operations(self):
        """Test basic cache operations."""
        cache = SearchResultCache()

        results = [{"text": "result1", "score": 0.9}]
        cache.set("查询", results)

        cached = cache.get("查询")
        assert cached == results

    def test_query_parameters(self):
        """Test different query parameters create different keys."""
        cache = SearchResultCache()

        results1 = [{"text": "result1"}]
        results2 = [{"text": "result2"}]

        cache.set("查询", results1, top_k=5)
        cache.set("查询", results2, top_k=10)

        assert cache.get("查询", top_k=5) == results1
        assert cache.get("查询", top_k=10) == results2

    def test_source_filter(self):
        """Test source filter affects cache key."""
        cache = SearchResultCache()

        results1 = [{"text": "from source1"}]
        results2 = [{"text": "from source2"}]

        cache.set("查询", results1, source_filter="source1")
        cache.set("查询", results2, source_filter="source2")

        assert cache.get("查询", source_filter="source1") == results1
        assert cache.get("查询", source_filter="source2") == results2


class TestResponseCache:
    """Tests for ResponseCache class."""

    def test_basic_operations(self):
        """Test basic cache operations."""
        cache = ResponseCache()

        response = {"answer": "测试回答", "confidence": 0.9}
        cache.set("问题", response)

        cached = cache.get("问题")
        assert cached == response

    def test_conversation_isolation(self):
        """Test responses are isolated by conversation."""
        cache = ResponseCache()

        response1 = {"answer": "回答1"}
        response2 = {"answer": "回答2"}

        cache.set("问题", response1, conversation_id="conv1")
        cache.set("问题", response2, conversation_id="conv2")

        assert cache.get("问题", conversation_id="conv1") == response1
        assert cache.get("问题", conversation_id="conv2") == response2


class TestCacheManager:
    """Tests for CacheManager class."""

    def test_initialization(self):
        """Test cache manager initialization."""
        manager = CacheManager()

        assert manager.embedding_cache is not None
        assert manager.search_cache is not None
        assert manager.response_cache is not None

    def test_get_all_stats(self):
        """Test getting all cache statistics."""
        manager = CacheManager()

        manager.embedding_cache.set("text", [0.1])
        manager.search_cache.set("query", [{"result": 1}])
        manager.response_cache.set("q", {"answer": "a"})

        stats = manager.get_all_stats()

        assert "embedding" in stats
        assert "search" in stats
        assert "response" in stats
        assert stats["embedding"]["size"] == 1

    def test_clear_all(self):
        """Test clearing all caches."""
        manager = CacheManager()

        manager.embedding_cache.set("text", [0.1])
        manager.search_cache.set("query", [])
        manager.response_cache.set("q", {})

        manager.clear_all()

        assert manager.embedding_cache._cache.size() == 0
        assert manager.search_cache._cache.size() == 0
        assert manager.response_cache._cache.size() == 0


class TestCachedDecorator:
    """Tests for cached decorator."""

    def test_basic_caching(self):
        """Test basic function caching."""
        cache: LRUCache[int] = LRUCache()
        call_count = 0

        @cached(cache, key_fn=lambda x: str(x))
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call (should be cached)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented

        # Different argument
        result3 = expensive_function(6)
        assert result3 == 12
        assert call_count == 2


class TestCachePerformance:
    """Performance tests for caching."""

    def test_embedding_cache_performance(self):
        """Test embedding cache handles large number of entries."""
        cache = EmbeddingCache(max_size=10000)

        # Add many entries
        for i in range(5000):
            cache.set(f"text_{i}", [float(i)])

        # Verify some entries
        assert cache.get("text_0") == [0.0]
        assert cache.get("text_4999") == [4999.0]

    def test_lru_eviction_performance(self):
        """Test LRU eviction handles many evictions efficiently."""
        cache: LRUCache[int] = LRUCache(max_size=100)

        # Add more entries than max_size
        for i in range(1000):
            cache.set(f"key_{i}", i)

        # Only last 100 should remain
        assert cache.size() == 100

        # Most recent should be accessible
        assert cache.get("key_999") == 999
        assert cache.get("key_900") == 900

        # Old entries should be evicted
        assert cache.get("key_0") is None

    def test_concurrent_access_safety(self):
        """Test cache is thread-safe under concurrent access."""
        import threading

        cache: LRUCache[int] = LRUCache(max_size=1000)
        errors = []

        def writer():
            try:
                for i in range(100):
                    cache.set(f"key_{threading.current_thread().name}_{i}", i)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    cache.get(f"key_{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, name=f"writer_{i}"))
            threads.append(threading.Thread(target=reader, name=f"reader_{i}"))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
