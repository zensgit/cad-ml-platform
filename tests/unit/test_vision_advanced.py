"""Unit tests for advanced vision features.

Tests caching, rate limiting, batch processing, and provider comparison.
"""

import asyncio

import pytest

from src.core.vision import (
    VisionDescription,
    VisionProviderError,
)
from src.core.vision.cache import (
    VisionCache,
    CachedVisionProvider,
    create_cached_provider,
)
from src.core.vision.rate_limiter import (
    TokenBucketRateLimiter,
    RateLimitedVisionProvider,
    RateLimitConfig,
    RateLimitError,
    create_rate_limited_provider,
)
from src.core.vision.batch import (
    BatchProcessor,
    process_images_batch,
)
from src.core.vision.comparison import (
    ProviderComparator,
    SelectionStrategy,
    compare_providers,
)


# Sample image data
SAMPLE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
)


class MockVisionProvider:
    """Mock provider for testing."""

    def __init__(
        self,
        name: str = "mock",
        confidence: float = 0.9,
        latency: float = 0.01,
        fail: bool = False,
    ):
        self.name = name
        self.confidence = confidence
        self.latency = latency
        self.fail = fail
        self.call_count = 0

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True
    ) -> VisionDescription:
        self.call_count += 1
        await asyncio.sleep(self.latency)

        if self.fail:
            raise VisionProviderError(self.name, "Simulated failure")

        return VisionDescription(
            summary=f"Summary from {self.name}",
            details=[f"Detail from {self.name}"],
            confidence=self.confidence,
        )

    @property
    def provider_name(self) -> str:
        return self.name


class TestVisionCache:
    """Tests for VisionCache."""

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = VisionCache(max_size=10)
        result = await cache.get(SAMPLE_PNG, "test", True)
        assert result is None
        assert cache.stats.misses == 1

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test cache hit returns stored value."""
        cache = VisionCache(max_size=10)

        # Store value
        description = VisionDescription(
            summary="Test", details=[], confidence=0.9
        )
        await cache.set(SAMPLE_PNG, "test", description, True)

        # Retrieve
        result = await cache.get(SAMPLE_PNG, "test", True)
        assert result is not None
        assert result.summary == "Test"
        assert cache.stats.hits == 1

    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test expired entries are not returned."""
        cache = VisionCache(max_size=10, ttl_seconds=0.01)

        description = VisionDescription(
            summary="Test", details=[], confidence=0.9
        )
        await cache.set(SAMPLE_PNG, "test", description, True)

        # Wait for expiration
        await asyncio.sleep(0.02)

        result = await cache.get(SAMPLE_PNG, "test", True)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = VisionCache(max_size=2)

        # Fill cache
        for i in range(3):
            img = bytes([i]) + SAMPLE_PNG
            desc = VisionDescription(
                summary=f"Test {i}", details=[], confidence=0.9
            )
            await cache.set(img, "test", desc, True)

        assert cache.stats.size == 2
        assert cache.stats.evictions >= 1

    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """Test clearing cache."""
        cache = VisionCache(max_size=10)

        description = VisionDescription(
            summary="Test", details=[], confidence=0.9
        )
        await cache.set(SAMPLE_PNG, "test", description, True)

        count = await cache.clear()
        assert count == 1
        assert cache.stats.size == 0


class TestCachedVisionProvider:
    """Tests for CachedVisionProvider."""

    @pytest.mark.asyncio
    async def test_caches_result(self):
        """Test that results are cached."""
        mock = MockVisionProvider()
        cached = CachedVisionProvider(mock, cache_max_size=10)

        # First call
        result1 = await cached.analyze_image(SAMPLE_PNG)
        assert mock.call_count == 1

        # Second call should hit cache
        result2 = await cached.analyze_image(SAMPLE_PNG)
        assert mock.call_count == 1  # No additional call
        assert result1.summary == result2.summary

    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistics."""
        mock = MockVisionProvider()
        cached = create_cached_provider(mock, max_size=10)

        await cached.analyze_image(SAMPLE_PNG)
        await cached.analyze_image(SAMPLE_PNG)

        stats = cached.cache_stats
        assert stats.hits == 1
        assert stats.misses == 1


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter."""

    @pytest.mark.asyncio
    async def test_allows_within_limit(self):
        """Test requests within limit are allowed."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=5)

        for _ in range(5):
            acquired = await limiter.acquire(wait=False)
            assert acquired is True

    @pytest.mark.asyncio
    async def test_rejects_over_limit(self):
        """Test requests over limit are rejected."""
        limiter = TokenBucketRateLimiter(rate=1.0, capacity=2)

        # Use all tokens
        await limiter.acquire(wait=False)
        await limiter.acquire(wait=False)

        # Should be rejected
        acquired = await limiter.acquire(wait=False)
        assert acquired is False

    @pytest.mark.asyncio
    async def test_refills_over_time(self):
        """Test tokens refill over time."""
        limiter = TokenBucketRateLimiter(rate=100.0, capacity=2)

        # Use all tokens
        await limiter.acquire(wait=False)
        await limiter.acquire(wait=False)

        # Wait for refill
        await asyncio.sleep(0.02)

        # Should have tokens now
        acquired = await limiter.acquire(wait=False)
        assert acquired is True


class TestRateLimitedVisionProvider:
    """Tests for RateLimitedVisionProvider."""

    @pytest.mark.asyncio
    async def test_allows_within_limit(self):
        """Test requests within limit succeed."""
        mock = MockVisionProvider()
        limited = RateLimitedVisionProvider(
            mock,
            config=RateLimitConfig(
                requests_per_minute=600,
                burst_size=10,
            ),
        )

        result = await limited.analyze_image(SAMPLE_PNG)
        assert result.summary == "Summary from mock"

    @pytest.mark.asyncio
    async def test_raises_when_limited(self):
        """Test raises error when rate limited."""
        mock = MockVisionProvider()
        limited = RateLimitedVisionProvider(
            mock,
            config=RateLimitConfig(
                requests_per_minute=60,
                burst_size=1,
            ),
        )

        # First request succeeds
        await limited.analyze_image(SAMPLE_PNG, wait_for_rate_limit=False)

        # Second should fail
        with pytest.raises(RateLimitError):
            await limited.analyze_image(SAMPLE_PNG, wait_for_rate_limit=False)


class TestBatchProcessor:
    """Tests for BatchProcessor."""

    @pytest.mark.asyncio
    async def test_processes_all_images(self):
        """Test all images are processed."""
        mock = MockVisionProvider()
        processor = BatchProcessor(mock, max_concurrency=3)

        images = [SAMPLE_PNG] * 5
        result = await processor.process_batch(images)

        assert result.total == 5
        assert result.completed == 5
        assert result.failed == 0
        assert mock.call_count == 5

    @pytest.mark.asyncio
    async def test_handles_failures(self):
        """Test failures are isolated."""
        mock = MockVisionProvider(fail=True)
        processor = BatchProcessor(mock, max_concurrency=2)

        images = [SAMPLE_PNG] * 3
        result = await processor.process_batch(images)

        assert result.total == 3
        assert result.completed == 0
        assert result.failed == 3
        assert len(result.errors) == 3

    @pytest.mark.asyncio
    async def test_progress_callback(self):
        """Test progress callback is called."""
        mock = MockVisionProvider()
        progress_updates = []

        def callback(progress):
            progress_updates.append(progress.progress_percent)

        result = await process_images_batch(
            mock,
            [SAMPLE_PNG] * 3,
            max_concurrency=1,
            progress_callback=callback,
        )

        assert result.completed == 3
        assert len(progress_updates) > 0


class TestProviderComparator:
    """Tests for ProviderComparator."""

    @pytest.mark.asyncio
    async def test_compares_multiple_providers(self):
        """Test comparing multiple providers."""
        providers = [
            MockVisionProvider("provider1", confidence=0.8),
            MockVisionProvider("provider2", confidence=0.9),
            MockVisionProvider("provider3", confidence=0.7),
        ]

        result = await compare_providers(SAMPLE_PNG, providers)

        assert result.providers_compared == 3
        assert result.success_count == 3
        assert "provider1" in result.provider_results
        assert "provider2" in result.provider_results
        assert "provider3" in result.provider_results

    @pytest.mark.asyncio
    async def test_selects_highest_confidence(self):
        """Test selects result with highest confidence."""
        providers = [
            MockVisionProvider("low", confidence=0.5),
            MockVisionProvider("high", confidence=0.95),
            MockVisionProvider("medium", confidence=0.7),
        ]

        comparator = ProviderComparator(
            providers,
            selection_strategy=SelectionStrategy.HIGHEST_CONFIDENCE,
        )
        result = await comparator.compare(SAMPLE_PNG)

        assert result.selected_provider == "high"
        assert result.selected_result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_handles_provider_failures(self):
        """Test handles failed providers gracefully."""
        providers = [
            MockVisionProvider("success", confidence=0.9),
            MockVisionProvider("failure", fail=True),
        ]

        result = await compare_providers(SAMPLE_PNG, providers)

        assert result.success_count == 1
        assert result.selected_provider == "success"
        assert "failure" in result.provider_results
        assert result.provider_results["failure"].success is False

    @pytest.mark.asyncio
    async def test_aggregated_summary(self):
        """Test generates aggregated summary."""
        providers = [
            MockVisionProvider("p1", confidence=0.9),
            MockVisionProvider("p2", confidence=0.8),
        ]

        result = await compare_providers(SAMPLE_PNG, providers)

        assert result.aggregated_summary is not None
        assert "p1" in result.aggregated_summary
        assert "p2" in result.aggregated_summary
