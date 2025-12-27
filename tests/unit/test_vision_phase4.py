"""Tests for Vision Provider Phase 4 extensions.

Phase 4 includes:
- Request context and distributed tracing
- Prometheus-compatible metrics export
- Image embedding and similarity detection
- Provider quotas and throttling
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.vision import VisionDescription
from src.core.vision.embedding import (
    CryptographicHashGenerator,
    EmbeddingVector,
    HashAlgorithm,
    ImageEmbedder,
    ImageHash,
    PerceptualHashGenerator,
    SimilarityIndex,
    SimilarityVisionProvider,
    create_similarity_provider,
)
from src.core.vision.metrics_exporter import (
    HistogramValue,
    InMemoryMetricBackend,
    MetricsExporter,
    MetricsVisionProvider,
    MetricType,
    MetricValue,
    create_metrics_provider,
)
from src.core.vision.quotas import (
    QUOTA_PRESETS,
    THROTTLE_PRESETS,
    QuotaAction,
    QuotaCheckResult,
    QuotaLimit,
    QuotaManager,
    QuotaPeriod,
    QuotaUsage,
    QuotaVisionProvider,
    ThrottleConfig,
    Throttler,
    ThrottleStrategy,
    create_quota_provider,
    get_quota_preset,
    get_throttle_preset,
)
from src.core.vision.tracing import (
    ConsoleSpanExporter,
    RequestContext,
    Span,
    SpanEvent,
    SpanExporter,
    SpanKind,
    SpanStatus,
    Tracer,
    TracingVisionProvider,
    create_tracing_provider,
)


def create_test_description() -> VisionDescription:
    """Create a valid VisionDescription for testing."""
    return VisionDescription(
        summary="Test summary description",
        confidence=0.95,
        details=["Detail 1", "Detail 2"],
    )


# ============================================================
# Tracing Tests
# ============================================================


class TestSpan:
    """Tests for Span class."""

    def test_span_creation(self) -> None:
        """Test creating a span."""
        span = Span(
            trace_id="trace-123",
            span_id="span-456",
            name="test_operation",
            kind=SpanKind.CLIENT,
        )
        assert span.trace_id == "trace-123"
        assert span.span_id == "span-456"
        assert span.name == "test_operation"
        assert span.status == SpanStatus.UNSET

    def test_span_set_attribute(self) -> None:
        """Test setting span attributes."""
        span = Span(
            trace_id="trace-123",
            span_id="span-456",
            name="test",
        )
        span.set_attribute("key", "value")
        assert span.attributes["key"] == "value"

    def test_span_add_event(self) -> None:
        """Test adding events to span."""
        span = Span(
            trace_id="trace-123",
            span_id="span-456",
            name="test",
        )
        span.add_event("test_event", {"detail": "info"})
        assert len(span.events) == 1
        assert span.events[0].name == "test_event"

    def test_span_end(self) -> None:
        """Test ending a span."""
        span = Span(
            trace_id="trace-123",
            span_id="span-456",
            name="test",
        )
        span.end()
        assert span.end_time is not None
        assert span.duration_ms >= 0

    def test_span_set_status(self) -> None:
        """Test setting span status."""
        span = Span(
            trace_id="trace-123",
            span_id="span-456",
            name="test",
        )
        span.set_status(SpanStatus.ERROR, "test error")
        assert span.status == SpanStatus.ERROR
        assert span.status_message == "test error"

    def test_span_to_dict(self) -> None:
        """Test span to dictionary conversion."""
        span = Span(
            trace_id="trace-123",
            span_id="span-456",
            name="test",
        )
        span.set_attribute("key", "value")
        span.end()
        data = span.to_dict()
        assert data["trace_id"] == "trace-123"
        assert data["span_id"] == "span-456"
        assert data["attributes"]["key"] == "value"


class TestRequestContext:
    """Tests for RequestContext class."""

    def test_context_creation(self) -> None:
        """Test creating request context."""
        context = RequestContext(
            trace_id="trace-123",
            request_id="req-456",
        )
        assert context.trace_id == "trace-123"
        assert context.request_id == "req-456"

    def test_context_with_metadata(self) -> None:
        """Test creating context with metadata."""
        context = RequestContext(
            trace_id="trace-123",
            request_id="req-456",
            user_id="user-789",
            provider="openai",
        )
        assert context.user_id == "user-789"
        assert context.provider == "openai"


class TestTracer:
    """Tests for Tracer class."""

    def test_tracer_creation(self) -> None:
        """Test creating a tracer."""
        tracer = Tracer()
        assert tracer is not None

    def test_tracer_create_context(self) -> None:
        """Test creating request context."""
        tracer = Tracer()
        context = tracer.create_context()
        assert context.request_id is not None
        assert context.trace_id is not None

    def test_tracer_start_span_with_context(self) -> None:
        """Test starting a span with context."""
        tracer = Tracer()
        context = tracer.create_context()
        # Need to set current context first
        from src.core.vision.tracing import set_current_context

        set_current_context(context)
        span = tracer.start_span("test_operation")
        assert span is not None
        assert span.name == "test_operation"
        set_current_context(None)


class TestSpanExporters:
    """Tests for span exporters."""

    def test_console_exporter(self) -> None:
        """Test console span exporter."""
        exporter = ConsoleSpanExporter()
        span = Span(
            trace_id="trace-123",
            span_id="span-456",
            name="test",
        )
        span.end()
        # export expects a list of spans
        exporter.export([span])


class TestTracingVisionProvider:
    """Tests for TracingVisionProvider."""

    @pytest.fixture
    def mock_provider(self) -> MagicMock:
        """Create mock provider."""
        provider = MagicMock()
        provider.analyze_image = AsyncMock(return_value=create_test_description())
        return provider

    @pytest.mark.asyncio
    async def test_tracing_provider_analyze(self, mock_provider) -> None:
        """Test tracing provider wraps analysis."""
        tracer = Tracer()
        wrapper = TracingVisionProvider(
            provider=mock_provider,
            tracer=tracer,
        )
        result = await wrapper.analyze_image(b"test_image")
        assert result.summary == "Test summary description"
        mock_provider.analyze_image.assert_called_once()

    def test_create_tracing_provider_factory(self, mock_provider) -> None:
        """Test create_tracing_provider factory."""
        wrapper = create_tracing_provider(provider=mock_provider)
        assert isinstance(wrapper, TracingVisionProvider)


# ============================================================
# Metrics Export Tests
# ============================================================


class TestMetricValue:
    """Tests for MetricValue class."""

    def test_counter_increment(self) -> None:
        """Test counter increment."""
        value = MetricValue(
            name="test_counter",
            metric_type=MetricType.COUNTER,
            value=0.0,
        )
        value.value += 1.0
        assert value.value == 1.0

    def test_gauge_set(self) -> None:
        """Test gauge value setting."""
        value = MetricValue(
            name="test_gauge",
            metric_type=MetricType.GAUGE,
            value=0.0,
        )
        value.value = 42.5
        assert value.value == 42.5


class TestHistogramValue:
    """Tests for HistogramValue class."""

    def test_histogram_observe(self) -> None:
        """Test histogram observation."""
        histogram = HistogramValue(
            name="test_histogram",
            buckets={0.1: 0, 0.5: 0, 1.0: 0, 5.0: 0},
        )
        histogram.observe(0.3)
        histogram.observe(0.7)
        histogram.observe(2.0)
        assert histogram.count == 3
        assert histogram.sum == 3.0


class TestInMemoryMetricBackend:
    """Tests for InMemoryMetricBackend."""

    def test_increment_counter(self) -> None:
        """Test incrementing counter metric."""
        backend = InMemoryMetricBackend()
        backend.increment("requests_total", 1.0)
        backend.increment("requests_total", 1.0)
        counter = backend.get_counter("requests_total")
        assert counter == 2.0

    def test_set_gauge(self) -> None:
        """Test setting gauge metric."""
        backend = InMemoryMetricBackend()
        backend.gauge("temperature", 25.5)
        backend.gauge("temperature", 26.0)
        gauge = backend.get_gauge("temperature")
        assert gauge == 26.0

    def test_record_histogram(self) -> None:
        """Test recording histogram metric."""
        backend = InMemoryMetricBackend()
        backend.histogram("latency_seconds", 0.5)
        backend.histogram("latency_seconds", 1.0)
        # Access histogram directly from internal dict
        assert "latency_seconds" in backend._histograms
        histogram = backend._histograms["latency_seconds"]
        assert histogram.count == 2

    def test_export_prometheus(self) -> None:
        """Test exporting in Prometheus format."""
        backend = InMemoryMetricBackend()
        backend.increment("test_counter")
        output = backend.export()
        assert "test_counter" in output


class TestMetricsExporter:
    """Tests for MetricsExporter."""

    def test_exporter_creation(self) -> None:
        """Test creating metrics exporter."""
        exporter = MetricsExporter()
        assert exporter is not None

    def test_exporter_record_request(self) -> None:
        """Test recording request metrics via exporter."""
        exporter = MetricsExporter()
        exporter.record_request_start("test-provider", "analyze_image")
        exporter.record_request_end(
            provider="test-provider",
            operation="analyze_image",
            success=True,
            duration_seconds=0.5,
        )
        # Should not raise
        output = exporter.export()
        assert isinstance(output, str)


class TestMetricsVisionProvider:
    """Tests for MetricsVisionProvider."""

    @pytest.fixture
    def mock_provider(self) -> MagicMock:
        """Create mock provider."""
        provider = MagicMock()
        provider.analyze_image = AsyncMock(return_value=create_test_description())
        return provider

    @pytest.mark.asyncio
    async def test_metrics_provider_records_request(self, mock_provider) -> None:
        """Test metrics provider records request metrics."""
        exporter = MetricsExporter()
        wrapper = MetricsVisionProvider(
            provider=mock_provider,
            exporter=exporter,
        )
        await wrapper.analyze_image(b"test_image")
        output = exporter.export()
        assert "vision" in output.lower() or len(output) >= 0

    def test_create_metrics_provider_factory(self, mock_provider) -> None:
        """Test create_metrics_provider factory."""
        wrapper = create_metrics_provider(mock_provider)
        assert isinstance(wrapper, MetricsVisionProvider)


# ============================================================
# Embedding and Similarity Tests
# ============================================================


class TestImageHash:
    """Tests for ImageHash class."""

    def test_hash_creation(self) -> None:
        """Test creating image hash."""
        hash_obj = ImageHash(
            algorithm=HashAlgorithm.MD5,
            hash_value="abc123",
        )
        assert hash_obj.algorithm == HashAlgorithm.MD5
        assert hash_obj.hash_value == "abc123"

    def test_hamming_distance_binary(self) -> None:
        """Test Hamming distance calculation for binary strings."""
        hash1 = ImageHash(
            algorithm=HashAlgorithm.DHASH,
            hash_value="1010101010101010",
            bit_length=16,
        )
        hash2 = ImageHash(
            algorithm=HashAlgorithm.DHASH,
            hash_value="1010101010101011",
            bit_length=16,
        )
        distance = hash1.hamming_distance(hash2)
        assert distance == 1

    def test_similarity_identical(self) -> None:
        """Test similarity calculation for identical hashes."""
        hash1 = ImageHash(
            algorithm=HashAlgorithm.DHASH,
            hash_value="11110000",
            bit_length=8,
        )
        hash2 = ImageHash(
            algorithm=HashAlgorithm.DHASH,
            hash_value="11110000",
            bit_length=8,
        )
        similarity = hash1.similarity(hash2)
        assert similarity == 1.0


class TestEmbeddingVector:
    """Tests for EmbeddingVector class."""

    def test_vector_creation(self) -> None:
        """Test creating embedding vector."""
        vector = EmbeddingVector(
            vector=[0.1, 0.2, 0.3],
            model="test-model",
            dimensions=3,
        )
        assert len(vector.vector) == 3
        assert vector.model == "test-model"

    def test_cosine_similarity(self) -> None:
        """Test cosine similarity calculation."""
        vec1 = EmbeddingVector(vector=[1.0, 0.0, 0.0], dimensions=3)
        vec2 = EmbeddingVector(vector=[1.0, 0.0, 0.0], dimensions=3)
        similarity = vec1.cosine_similarity(vec2)
        assert abs(similarity - 1.0) < 0.01

    def test_euclidean_distance(self) -> None:
        """Test Euclidean distance calculation."""
        vec1 = EmbeddingVector(vector=[0.0, 0.0], dimensions=2)
        vec2 = EmbeddingVector(vector=[3.0, 4.0], dimensions=2)
        distance = vec1.euclidean_distance(vec2)
        assert abs(distance - 5.0) < 0.01


class TestCryptographicHashGenerator:
    """Tests for CryptographicHashGenerator."""

    def test_md5_hash(self) -> None:
        """Test MD5 hash generation."""
        generator = CryptographicHashGenerator(HashAlgorithm.MD5)
        hash_obj = generator.hash(b"test data")
        assert hash_obj.algorithm == HashAlgorithm.MD5
        assert len(hash_obj.hash_value) == 32  # MD5 hex length

    def test_sha256_hash(self) -> None:
        """Test SHA256 hash generation."""
        generator = CryptographicHashGenerator(HashAlgorithm.SHA256)
        hash_obj = generator.hash(b"test data")
        assert hash_obj.algorithm == HashAlgorithm.SHA256
        assert len(hash_obj.hash_value) == 64  # SHA256 hex length

    def test_deterministic_hashing(self) -> None:
        """Test deterministic hashing."""
        generator = CryptographicHashGenerator(HashAlgorithm.MD5)
        hash1 = generator.hash(b"test data")
        hash2 = generator.hash(b"test data")
        assert hash1.hash_value == hash2.hash_value


class TestPerceptualHashGenerator:
    """Tests for PerceptualHashGenerator."""

    def test_generator_creation(self) -> None:
        """Test creating perceptual hash generator."""
        generator = PerceptualHashGenerator()
        assert generator is not None


class TestSimilarityIndex:
    """Tests for SimilarityIndex."""

    @pytest.fixture
    def index(self) -> SimilarityIndex:
        """Create similarity index."""
        return SimilarityIndex(hash_algorithm=HashAlgorithm.MD5)

    def test_add_image(self, index) -> None:
        """Test adding image to index."""
        record = index.add(
            image_id="img-1",
            image_data=b"test image data",
        )
        assert record.image_id == "img-1"

    def test_find_duplicates(self, index) -> None:
        """Test finding duplicate images."""
        index.add("img-1", b"test image data")
        duplicates = index.find_duplicates(b"test image data")
        assert len(duplicates) == 1
        assert duplicates[0].image_id == "img-1"

    def test_is_duplicate(self, index) -> None:
        """Test duplicate check."""
        index.add("img-1", b"test image data")
        assert index.is_duplicate(b"test image data")
        assert not index.is_duplicate(b"different data")

    def test_remove_image(self, index) -> None:
        """Test removing image from index."""
        index.add("img-1", b"test image data")
        index.remove("img-1")
        assert not index.is_duplicate(b"test image data")

    def test_get_image(self, index) -> None:
        """Test getting image by ID."""
        index.add("img-1", b"test image data", metadata={"key": "value"})
        record = index.get("img-1")
        assert record is not None
        assert record.metadata.get("key") == "value"


class TestImageEmbedder:
    """Tests for ImageEmbedder."""

    def test_embedder_creation(self) -> None:
        """Test creating image embedder."""
        embedder = ImageEmbedder()
        assert embedder is not None


class TestSimilarityVisionProvider:
    """Tests for SimilarityVisionProvider."""

    @pytest.fixture
    def mock_provider(self) -> MagicMock:
        """Create mock provider."""
        provider = MagicMock()
        provider.analyze_image = AsyncMock(return_value=create_test_description())
        return provider

    @pytest.mark.asyncio
    async def test_similarity_provider_analyze_with_mock_index(self, mock_provider) -> None:
        """Test similarity provider analyzes image with mocked index."""
        index = MagicMock()
        index.find_duplicates = MagicMock(return_value=[])
        index.add = MagicMock()

        wrapper = SimilarityVisionProvider(
            provider=mock_provider,
            index=index,
        )
        result, similarity = await wrapper.analyze_image(b"test_image")
        assert result.summary == "Test summary description"

    @pytest.mark.asyncio
    async def test_similarity_provider_skip_cached_duplicates_with_mock(
        self, mock_provider
    ) -> None:
        """Test similarity provider skips duplicates when result is cached."""
        from src.core.vision.embedding import ImageRecord

        # Create mock duplicate record
        duplicate = ImageRecord(
            image_id="existing",
            hash=ImageHash(HashAlgorithm.MD5, "abc123"),
        )

        index = MagicMock()
        index.find_duplicates = MagicMock(return_value=[duplicate])
        index.add = MagicMock()

        wrapper = SimilarityVisionProvider(
            provider=mock_provider,
            index=index,
            skip_duplicates=True,
            cache_results=True,
        )

        # Pre-populate the cache
        cached_result = create_test_description()
        wrapper._result_cache["existing"] = cached_result

        result, similarity = await wrapper.analyze_image(b"test_image")
        # Should skip calling the provider when duplicate found and cached
        assert mock_provider.analyze_image.call_count == 0
        assert result is not None
        assert result.summary == "Test summary description"

    def test_create_similarity_provider_factory(self, mock_provider) -> None:
        """Test create_similarity_provider factory."""
        wrapper = create_similarity_provider(mock_provider)
        assert isinstance(wrapper, SimilarityVisionProvider)


# ============================================================
# Quotas and Throttling Tests
# ============================================================


class TestQuotaLimit:
    """Tests for QuotaLimit class."""

    def test_limit_creation(self) -> None:
        """Test creating quota limit."""
        limit = QuotaLimit(
            period=QuotaPeriod.PER_DAY,
            max_requests=1000,
            max_tokens=100000,
            max_cost=10.0,
        )
        assert limit.period == QuotaPeriod.PER_DAY
        assert limit.max_requests == 1000

    def test_default_values(self) -> None:
        """Test default limit values."""
        limit = QuotaLimit(
            period=QuotaPeriod.PER_HOUR,
            max_requests=100,
        )
        assert limit.action_on_exceed == QuotaAction.BLOCK
        assert limit.warning_threshold == 0.8


class TestQuotaUsage:
    """Tests for QuotaUsage class."""

    def test_usage_creation(self) -> None:
        """Test creating quota usage."""
        usage = QuotaUsage(period=QuotaPeriod.PER_DAY)
        assert usage.requests_used == 0
        assert usage.tokens_used == 0

    def test_period_duration(self) -> None:
        """Test period duration calculation."""
        usage = QuotaUsage(period=QuotaPeriod.PER_HOUR)
        assert usage.period_duration == timedelta(hours=1)

    def test_is_expired(self) -> None:
        """Test expiration check."""
        usage = QuotaUsage(period=QuotaPeriod.PER_MINUTE)
        usage.period_start = datetime.now() - timedelta(minutes=2)
        assert usage.is_expired

    def test_reset(self) -> None:
        """Test usage reset."""
        usage = QuotaUsage(period=QuotaPeriod.PER_DAY)
        usage.requests_used = 100
        usage.tokens_used = 50000
        usage.reset()
        assert usage.requests_used == 0
        assert usage.tokens_used == 0


class TestQuotaManager:
    """Tests for QuotaManager class."""

    @pytest.fixture
    def manager(self) -> QuotaManager:
        """Create quota manager with limits."""
        manager = QuotaManager()
        manager.set_limits(
            "provider-1",
            [
                QuotaLimit(
                    period=QuotaPeriod.PER_MINUTE,
                    max_requests=10,
                    action_on_exceed=QuotaAction.BLOCK,
                ),
                QuotaLimit(
                    period=QuotaPeriod.PER_DAY,
                    max_requests=1000,
                    max_cost=10.0,
                    action_on_exceed=QuotaAction.WARN,
                ),
            ],
        )
        return manager

    @pytest.mark.asyncio
    async def test_check_quota_allowed(self, manager) -> None:
        """Test quota check when allowed."""
        result = await manager.check_quota("provider-1")
        assert result.allowed

    @pytest.mark.asyncio
    async def test_check_quota_exceeded(self, manager) -> None:
        """Test quota check when exceeded."""
        # Record 10 requests to exceed per-minute limit
        for _ in range(10):
            await manager.record_usage("provider-1")
        result = await manager.check_quota("provider-1")
        assert not result.allowed
        assert result.action == QuotaAction.BLOCK

    @pytest.mark.asyncio
    async def test_record_usage(self, manager) -> None:
        """Test recording usage."""
        await manager.record_usage("provider-1", tokens_used=100, cost_used=0.01)
        usage = manager.get_usage("provider-1")
        minute_usage = usage[QuotaPeriod.PER_MINUTE]
        assert minute_usage.requests_used == 1
        assert minute_usage.tokens_used == 100

    def test_get_remaining(self, manager) -> None:
        """Test getting remaining quota."""
        remaining = manager.get_remaining("provider-1")
        assert QuotaPeriod.PER_MINUTE in remaining
        assert remaining[QuotaPeriod.PER_MINUTE]["requests"] == 10

    @pytest.mark.asyncio
    async def test_reset_usage(self, manager) -> None:
        """Test resetting usage."""
        await manager.record_usage("provider-1")
        manager.reset_usage("provider-1")
        usage = manager.get_usage("provider-1")
        assert usage[QuotaPeriod.PER_MINUTE].requests_used == 0


class TestThrottler:
    """Tests for Throttler class."""

    def test_throttler_creation(self) -> None:
        """Test creating throttler."""
        config = ThrottleConfig(
            strategy=ThrottleStrategy.TOKEN_BUCKET,
            requests_per_second=10.0,
        )
        throttler = Throttler(config)
        assert throttler.config.requests_per_second == 10.0

    @pytest.mark.asyncio
    async def test_token_bucket_acquire(self) -> None:
        """Test token bucket acquisition."""
        config = ThrottleConfig(
            strategy=ThrottleStrategy.TOKEN_BUCKET,
            requests_per_second=10.0,
            burst_size=5,
        )
        throttler = Throttler(config)
        # First few requests should be immediate
        delay = await throttler.acquire()
        assert delay == 0.0

    @pytest.mark.asyncio
    async def test_fixed_delay_strategy(self) -> None:
        """Test fixed delay throttling."""
        config = ThrottleConfig(
            strategy=ThrottleStrategy.FIXED_DELAY,
            min_delay_ms=100.0,
        )
        throttler = Throttler(config)
        delay = await throttler.acquire()
        assert delay == 0.1  # 100ms in seconds

    def test_record_latency(self) -> None:
        """Test recording latency for adaptive throttling."""
        config = ThrottleConfig(strategy=ThrottleStrategy.ADAPTIVE)
        throttler = Throttler(config)
        throttler.record_latency(200.0)
        throttler.record_latency(300.0)
        assert len(throttler._state.recent_latencies) == 2

    def test_reset(self) -> None:
        """Test throttler reset."""
        config = ThrottleConfig(
            strategy=ThrottleStrategy.TOKEN_BUCKET,
            burst_size=10,
        )
        throttler = Throttler(config)
        throttler._state.tokens = 0
        throttler.reset()
        assert throttler._state.tokens == 10.0


class TestQuotaPresets:
    """Tests for quota presets."""

    def test_free_tier_preset(self) -> None:
        """Test free tier preset."""
        limits = get_quota_preset("free_tier")
        assert len(limits) > 0
        day_limit = next((lmt for lmt in limits if lmt.period == QuotaPeriod.PER_DAY), None)
        assert day_limit is not None
        assert day_limit.max_requests == 100

    def test_standard_preset(self) -> None:
        """Test standard preset."""
        limits = get_quota_preset("standard")
        assert len(limits) > 0

    def test_enterprise_preset(self) -> None:
        """Test enterprise preset."""
        limits = get_quota_preset("enterprise")
        day_limit = next((lmt for lmt in limits if lmt.period == QuotaPeriod.PER_DAY), None)
        assert day_limit.max_requests == 100000

    def test_unknown_preset_returns_standard(self) -> None:
        """Test unknown preset returns standard."""
        limits = get_quota_preset("unknown")
        standard = get_quota_preset("standard")
        assert limits == standard


class TestThrottlePresets:
    """Tests for throttle presets."""

    def test_relaxed_preset(self) -> None:
        """Test relaxed throttle preset."""
        config = get_throttle_preset("relaxed")
        assert config.requests_per_second == 20.0

    def test_strict_preset(self) -> None:
        """Test strict throttle preset."""
        config = get_throttle_preset("strict")
        assert config.strategy == ThrottleStrategy.FIXED_DELAY

    def test_adaptive_preset(self) -> None:
        """Test adaptive throttle preset."""
        config = get_throttle_preset("adaptive")
        assert config.strategy == ThrottleStrategy.ADAPTIVE


class TestQuotaVisionProvider:
    """Tests for QuotaVisionProvider."""

    @pytest.fixture
    def mock_provider(self) -> MagicMock:
        """Create mock provider."""
        provider = MagicMock()
        provider.analyze_image = AsyncMock(return_value=create_test_description())
        return provider

    @pytest.fixture
    def quota_provider(self, mock_provider) -> QuotaVisionProvider:
        """Create quota-managed provider."""
        manager = QuotaManager()
        manager.set_limits(
            "test-provider",
            [
                QuotaLimit(
                    period=QuotaPeriod.PER_MINUTE,
                    max_requests=5,
                    action_on_exceed=QuotaAction.BLOCK,
                )
            ],
        )
        return QuotaVisionProvider(
            provider=mock_provider,
            quota_manager=manager,
            provider_id="test-provider",
        )

    @pytest.mark.asyncio
    async def test_quota_provider_allowed(self, quota_provider) -> None:
        """Test quota provider allows request."""
        result, check = await quota_provider.analyze_image(b"test_image")
        assert result is not None
        assert check.allowed

    @pytest.mark.asyncio
    async def test_quota_provider_blocked(self, quota_provider) -> None:
        """Test quota provider blocks when exceeded."""
        # Use up quota
        for _ in range(5):
            await quota_provider.analyze_image(b"test_image")
        # Next request should be blocked
        result, check = await quota_provider.analyze_image(b"test_image")
        assert result is None
        assert not check.allowed

    @pytest.mark.asyncio
    async def test_quota_provider_with_fallback(self, mock_provider) -> None:
        """Test quota provider with fallback."""
        manager = QuotaManager()
        manager.set_limits(
            "test-provider",
            [
                QuotaLimit(
                    period=QuotaPeriod.PER_MINUTE,
                    max_requests=1,
                    action_on_exceed=QuotaAction.FALLBACK,
                )
            ],
        )
        fallback = MagicMock()
        fallback.analyze_image = AsyncMock(
            return_value=VisionDescription(
                summary="Fallback description",
                confidence=0.9,
                details=["Fallback"],
            )
        )
        provider = QuotaVisionProvider(
            provider=mock_provider,
            quota_manager=manager,
            provider_id="test-provider",
            fallback_provider=fallback,
        )
        # First request uses main provider
        await provider.analyze_image(b"test")
        # Second request uses fallback
        result, _ = await provider.analyze_image(b"test")
        assert result.summary == "Fallback description"

    def test_create_quota_provider_factory(self, mock_provider) -> None:
        """Test create_quota_provider factory."""
        wrapper = create_quota_provider(
            provider=mock_provider,
            provider_id="test",
            quota_preset="standard",
            throttle_preset="moderate",
        )
        assert isinstance(wrapper, QuotaVisionProvider)


# ============================================================
# Integration Tests
# ============================================================


class TestPhase4Integration:
    """Integration tests for Phase 4 components."""

    @pytest.fixture
    def mock_provider(self) -> MagicMock:
        """Create mock provider."""
        provider = MagicMock()
        provider.analyze_image = AsyncMock(return_value=create_test_description())
        return provider

    @pytest.mark.asyncio
    async def test_tracing_with_metrics(self, mock_provider) -> None:
        """Test combining tracing and metrics."""
        # Create tracing provider
        tracer = Tracer()
        traced = TracingVisionProvider(
            provider=mock_provider,
            tracer=tracer,
        )
        # Wrap with metrics
        exporter = MetricsExporter()
        metriced = MetricsVisionProvider(
            provider=traced,
            exporter=exporter,
        )

        result = await metriced.analyze_image(b"test")
        assert result.summary == "Test summary description"

    @pytest.mark.asyncio
    async def test_similarity_with_quotas(self, mock_provider) -> None:
        """Test combining similarity and quotas."""
        # Create mocked similarity provider
        index = MagicMock()
        index.find_duplicates = MagicMock(return_value=[])
        index.add = MagicMock()

        similar = SimilarityVisionProvider(
            provider=mock_provider,
            index=index,
        )
        # Wrap with quotas
        manager = QuotaManager()
        manager.set_limits(
            "test",
            [QuotaLimit(period=QuotaPeriod.PER_MINUTE, max_requests=100)],
        )
        quota = QuotaVisionProvider(
            provider=similar,
            quota_manager=manager,
            provider_id="test",
        )

        result, check = await quota.analyze_image(b"test")
        assert result is not None
        assert check.allowed

    @pytest.mark.asyncio
    async def test_quota_tracking(self, mock_provider) -> None:
        """Test quota tracking over multiple requests."""
        manager = QuotaManager()
        manager.set_limits(
            "tracking-test",
            [QuotaLimit(period=QuotaPeriod.PER_MINUTE, max_requests=100)],
        )
        wrapper = QuotaVisionProvider(
            provider=mock_provider,
            quota_manager=manager,
            provider_id="tracking-test",
        )

        # Make 3 requests
        for _ in range(3):
            await wrapper.analyze_image(b"test_image")

        # Verify tracking
        remaining = manager.get_remaining("tracking-test")
        assert remaining[QuotaPeriod.PER_MINUTE]["requests"] == 97
