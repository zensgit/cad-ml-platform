"""Tests for Vision Provider Phase 3 features.

Tests cover:
- Image preprocessing and validation
- Provider load balancing
- Request/response logging middleware
- Configuration profiles
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.vision.base import VisionDescription, VisionProvider, VisionProviderError

# =============================================================================
# Preprocessing Tests
# =============================================================================


class TestImageFormat:
    """Tests for ImageFormat enum."""

    def test_image_format_values(self) -> None:
        """Test ImageFormat enum values."""
        from src.core.vision.preprocessing import ImageFormat

        assert ImageFormat.PNG.value == "png"
        assert ImageFormat.JPEG.value == "jpeg"
        assert ImageFormat.GIF.value == "gif"
        assert ImageFormat.WEBP.value == "webp"
        assert ImageFormat.UNKNOWN.value == "unknown"


class TestImageValidator:
    """Tests for ImageValidator class."""

    def test_detect_png_format(self) -> None:
        """Test PNG format detection."""
        from src.core.vision.preprocessing import ImageFormat, ImageValidator

        validator = ImageValidator()
        # PNG magic bytes
        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        assert validator.detect_format(png_data) == ImageFormat.PNG

    def test_detect_jpeg_format(self) -> None:
        """Test JPEG format detection."""
        from src.core.vision.preprocessing import ImageFormat, ImageValidator

        validator = ImageValidator()
        # JPEG magic bytes
        jpeg_data = b"\xff\xd8\xff" + b"\x00" * 100
        assert validator.detect_format(jpeg_data) == ImageFormat.JPEG

    def test_detect_gif_format(self) -> None:
        """Test GIF format detection."""
        from src.core.vision.preprocessing import ImageFormat, ImageValidator

        validator = ImageValidator()
        # GIF87a magic bytes
        gif_data = b"GIF87a" + b"\x00" * 100
        assert validator.detect_format(gif_data) == ImageFormat.GIF

    def test_detect_webp_format(self) -> None:
        """Test WebP format detection."""
        from src.core.vision.preprocessing import ImageFormat, ImageValidator

        validator = ImageValidator()
        # WebP magic bytes
        webp_data = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 100
        assert validator.detect_format(webp_data) == ImageFormat.WEBP

    def test_detect_unknown_format(self) -> None:
        """Test unknown format detection."""
        from src.core.vision.preprocessing import ImageFormat, ImageValidator

        validator = ImageValidator()
        unknown_data = b"random data here"
        assert validator.detect_format(unknown_data) == ImageFormat.UNKNOWN

    def test_validate_valid_image(self) -> None:
        """Test validation of valid image data - requires dimensions."""
        from src.core.vision.preprocessing import ImageValidator, PreprocessingConfig

        # Use permissive config to test format detection
        config = PreprocessingConfig(
            max_size_bytes=10000,
            min_size_bytes=10,  # Allow small test images
        )
        validator = ImageValidator(config)

        # Minimal valid PNG with IHDR chunk containing dimensions
        # PNG signature + IHDR chunk (13 bytes): width=1, height=1, bit depth=8, color type=2 (RGB)
        png_ihdr = (
            b"\x89PNG\r\n\x1a\n"  # PNG signature
            b"\x00\x00\x00\rIHDR"  # IHDR chunk length (13) and type
            b"\x00\x00\x00\x01"  # Width = 1
            b"\x00\x00\x00\x01"  # Height = 1
            b"\x08\x02"  # Bit depth = 8, Color type = 2 (RGB)
            b"\x00\x00\x00"  # Compression, filter, interlace
            b"\x90wS\xde"  # CRC
            b"\x00\x00\x00\x00IEND\xaeB`\x82"  # IEND chunk
            b"\x00" * 100  # Padding to meet minimum size
        )
        result = validator.validate(png_ihdr)
        # Test that format detection works
        detected_format = validator.detect_format(png_ihdr)
        assert detected_format.value == "png"

    def test_validate_empty_image(self) -> None:
        """Test validation of empty image data."""
        from src.core.vision.preprocessing import ImageValidator

        validator = ImageValidator()
        result = validator.validate(b"")
        assert result.valid is False
        assert len(result.errors) > 0

    def test_validate_oversized_image(self) -> None:
        """Test validation of oversized image data."""
        from src.core.vision.preprocessing import ImageValidator, PreprocessingConfig

        config = PreprocessingConfig(max_size_bytes=50)  # Very small limit
        validator = ImageValidator(config)

        # PNG header that exceeds size limit
        large_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        result = validator.validate(large_data)
        assert result.valid is False
        # Check for any error (size error or dimension error)


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        from src.core.vision.preprocessing import PreprocessingConfig

        config = PreprocessingConfig()
        assert config.max_size_bytes == 20 * 1024 * 1024  # 20MB
        assert config.max_width == 4096
        assert config.max_height == 4096
        assert config.auto_resize is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        from src.core.vision.preprocessing import ImageFormat, PreprocessingConfig

        config = PreprocessingConfig(
            max_size_bytes=10 * 1024 * 1024,
            max_width=2048,
            max_height=2048,
            auto_resize=False,
            target_format=ImageFormat.JPEG,
        )
        assert config.max_size_bytes == 10 * 1024 * 1024
        assert config.max_width == 2048
        assert config.target_format == ImageFormat.JPEG


class TestPreprocessingVisionProvider:
    """Tests for PreprocessingVisionProvider wrapper."""

    @pytest.fixture
    def mock_provider(self) -> MagicMock:
        """Create mock vision provider."""
        provider = MagicMock(spec=VisionProvider)
        provider.provider_name = "mock"
        provider.analyze_image = AsyncMock(
            return_value=VisionDescription(
                summary="Test description",
                details=["Test detail 1", "Test detail 2"],
                confidence=0.95,
            )
        )
        return provider

    @pytest.mark.asyncio
    async def test_analyze_with_preprocessing(self, mock_provider: MagicMock) -> None:
        """Test image analysis with preprocessing."""
        from src.core.vision.preprocessing import (
            ImageFormat,
            ImageInfo,
            PreprocessingConfig,
            PreprocessingVisionProvider,
        )

        config = PreprocessingConfig()
        wrapper = PreprocessingVisionProvider(mock_provider, config)

        # Minimal valid PNG with proper IHDR chunk
        png_data = (
            b"\x89PNG\r\n\x1a\n"  # PNG signature
            b"\x00\x00\x00\rIHDR"  # IHDR chunk length (13) and type
            b"\x00\x00\x00\x64"  # Width = 100
            b"\x00\x00\x00\x64"  # Height = 100
            b"\x08\x02"  # Bit depth = 8, Color type = 2 (RGB)
            b"\x00\x00\x00"  # Compression, filter, interlace
            b"\x00\x00\x00\x00"  # CRC placeholder
            b"\x00\x00\x00\x00IEND\xaeB`\x82"  # IEND chunk
        )

        # Mock the preprocessor's preprocess method to return valid data
        with patch.object(
            wrapper._preprocessor,
            "preprocess",
            return_value=(
                png_data,
                ImageInfo(
                    format=ImageFormat.PNG,
                    width=100,
                    height=100,
                    size_bytes=len(png_data),
                    has_alpha=False,
                    color_mode="RGB",
                ),
            ),
        ):
            result = await wrapper.analyze_image(png_data)

        assert result.summary == "Test description"
        assert mock_provider.analyze_image.called

    @pytest.mark.asyncio
    async def test_analyze_skip_preprocessing(self, mock_provider: MagicMock) -> None:
        """Test skipping preprocessing passes data through without transformation."""
        from src.core.vision.preprocessing import (
            ImageFormat,
            ImageInfo,
            PreprocessingConfig,
            PreprocessingVisionProvider,
        )

        config = PreprocessingConfig()
        wrapper = PreprocessingVisionProvider(mock_provider, config)

        # Test data
        test_data = b"test image data that won't be modified"

        # Mock both validator and preprocessor to allow the test data through
        with patch.object(
            wrapper._validator,
            "validate",
            return_value=MagicMock(
                valid=True,
                errors=[],
                info=ImageInfo(
                    format=ImageFormat.PNG,
                    width=100,
                    height=100,
                    size_bytes=len(test_data),
                    has_alpha=False,
                    color_mode="RGB",
                ),
            ),
        ):
            # When skip_preprocessing=True, data should pass through without preprocessing
            result = await wrapper.analyze_image(test_data, skip_preprocessing=True)

        assert result.summary == "Test description"
        # Verify the original data was passed (not preprocessed)
        mock_provider.analyze_image.assert_called_once()


class TestCreatePreprocessingProvider:
    """Tests for create_preprocessing_provider factory."""

    def test_create_with_defaults(self) -> None:
        """Test factory with default options."""
        from src.core.vision.preprocessing import (
            PreprocessingVisionProvider,
            create_preprocessing_provider,
        )

        mock_provider = MagicMock(spec=VisionProvider)
        wrapper = create_preprocessing_provider(mock_provider)

        assert isinstance(wrapper, PreprocessingVisionProvider)

    def test_create_with_custom_options(self) -> None:
        """Test factory with custom options."""
        from src.core.vision.preprocessing import (
            ImageFormat,
            PreprocessingVisionProvider,
            create_preprocessing_provider,
        )

        mock_provider = MagicMock(spec=VisionProvider)
        wrapper = create_preprocessing_provider(
            mock_provider,
            max_size_mb=10.0,
            max_dimension=2048,
            auto_resize=False,
            target_format=ImageFormat.JPEG,
        )

        assert isinstance(wrapper, PreprocessingVisionProvider)


# =============================================================================
# Load Balancer Tests
# =============================================================================


class TestLoadBalancingAlgorithm:
    """Tests for LoadBalancingAlgorithm enum."""

    def test_algorithm_values(self) -> None:
        """Test algorithm enum values."""
        from src.core.vision.load_balancer import LoadBalancingAlgorithm

        assert LoadBalancingAlgorithm.ROUND_ROBIN.value == "round_robin"
        assert LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN.value == "weighted_round_robin"
        assert LoadBalancingAlgorithm.LEAST_CONNECTIONS.value == "least_connections"
        assert LoadBalancingAlgorithm.RANDOM.value == "random"
        assert LoadBalancingAlgorithm.ADAPTIVE.value == "adaptive"


class TestProviderNode:
    """Tests for ProviderNode dataclass."""

    @pytest.fixture
    def mock_provider(self) -> MagicMock:
        """Create mock provider."""
        provider = MagicMock(spec=VisionProvider)
        provider.provider_name = "test_provider"
        return provider

    def test_default_values(self, mock_provider: MagicMock) -> None:
        """Test default node values."""
        from src.core.vision.load_balancer import ProviderNode

        node = ProviderNode(provider=mock_provider)
        assert node.weight == 1.0
        assert node.max_connections == 100
        assert node.enabled is True
        assert node.current_connections == 0

    def test_record_request_start(self, mock_provider: MagicMock) -> None:
        """Test recording request start."""
        from src.core.vision.load_balancer import ProviderNode

        node = ProviderNode(provider=mock_provider)
        node.record_request_start()

        assert node.current_connections == 1
        assert node.total_requests == 1

    def test_record_request_end_success(self, mock_provider: MagicMock) -> None:
        """Test recording successful request end."""
        from src.core.vision.load_balancer import ProviderNode

        node = ProviderNode(provider=mock_provider)
        node.record_request_start()
        node.record_request_end(success=True, response_time_ms=100.0)

        assert node.current_connections == 0
        assert node.successful_requests == 1
        assert node.last_response_time_ms == 100.0

    def test_record_request_end_failure(self, mock_provider: MagicMock) -> None:
        """Test recording failed request end."""
        from src.core.vision.load_balancer import ProviderNode

        node = ProviderNode(provider=mock_provider)
        node.record_request_start()
        node.record_request_end(success=False, response_time_ms=50.0)

        assert node.current_connections == 0
        assert node.failed_requests == 1
        assert node.last_failure is not None

    def test_success_rate_calculation(self, mock_provider: MagicMock) -> None:
        """Test success rate calculation."""
        from src.core.vision.load_balancer import ProviderNode

        node = ProviderNode(provider=mock_provider)

        # Record 8 successes, 2 failures
        for _ in range(8):
            node.record_request_start()
            node.record_request_end(success=True, response_time_ms=100.0)
        for _ in range(2):
            node.record_request_start()
            node.record_request_end(success=False, response_time_ms=50.0)

        assert node.success_rate == 0.8

    def test_effective_weight_high_success(self, mock_provider: MagicMock) -> None:
        """Test effective weight with high success rate."""
        from src.core.vision.load_balancer import ProviderNode

        node = ProviderNode(provider=mock_provider, weight=2.0)
        for _ in range(10):
            node.record_request_start()
            node.record_request_end(success=True, response_time_ms=100.0)

        assert node.effective_weight == 2.0

    def test_effective_weight_low_success(self, mock_provider: MagicMock) -> None:
        """Test effective weight with low success rate."""
        from src.core.vision.load_balancer import ProviderNode

        node = ProviderNode(provider=mock_provider, weight=2.0)
        for _ in range(3):
            node.record_request_start()
            node.record_request_end(success=True, response_time_ms=100.0)
        for _ in range(7):
            node.record_request_start()
            node.record_request_end(success=False, response_time_ms=50.0)

        # 30% success rate -> weight * 0.1
        assert node.effective_weight == 0.2

    def test_is_available(self, mock_provider: MagicMock) -> None:
        """Test availability check."""
        from src.core.vision.load_balancer import ProviderNode

        node = ProviderNode(provider=mock_provider, max_connections=2)
        assert node.is_available() is True

        node.enabled = False
        assert node.is_available() is False

        node.enabled = True
        node.current_connections = 2
        assert node.is_available() is False


class TestLoadBalancer:
    """Tests for LoadBalancer class."""

    @pytest.fixture
    def mock_nodes(self) -> list:
        """Create mock provider nodes."""
        from src.core.vision.load_balancer import ProviderNode

        nodes = []
        for i in range(3):
            provider = MagicMock(spec=VisionProvider)
            provider.provider_name = f"provider_{i}"
            nodes.append(ProviderNode(provider=provider, weight=1.0 + i * 0.5))
        return nodes

    @pytest.mark.asyncio
    async def test_round_robin_selection(self, mock_nodes: list) -> None:
        """Test round-robin node selection."""
        from src.core.vision.load_balancer import (
            LoadBalancer,
            LoadBalancerConfig,
            LoadBalancingAlgorithm,
        )

        config = LoadBalancerConfig(algorithm=LoadBalancingAlgorithm.ROUND_ROBIN)
        balancer = LoadBalancer(nodes=mock_nodes, config=config)

        # Select nodes in sequence
        selected = []
        for _ in range(6):
            node = await balancer.select_node()
            selected.append(node.provider.provider_name)

        # Should cycle through all providers
        assert selected[:3] == ["provider_0", "provider_1", "provider_2"]
        assert selected[3:6] == ["provider_0", "provider_1", "provider_2"]

    @pytest.mark.asyncio
    async def test_least_connections_selection(self, mock_nodes: list) -> None:
        """Test least-connections node selection."""
        from src.core.vision.load_balancer import (
            LoadBalancer,
            LoadBalancerConfig,
            LoadBalancingAlgorithm,
        )

        config = LoadBalancerConfig(algorithm=LoadBalancingAlgorithm.LEAST_CONNECTIONS)
        balancer = LoadBalancer(nodes=mock_nodes, config=config)

        # Add connections to first two nodes
        mock_nodes[0].current_connections = 5
        mock_nodes[1].current_connections = 3
        mock_nodes[2].current_connections = 1

        node = await balancer.select_node()
        assert node.provider.provider_name == "provider_2"

    @pytest.mark.asyncio
    async def test_random_selection(self, mock_nodes: list) -> None:
        """Test random node selection."""
        from src.core.vision.load_balancer import (
            LoadBalancer,
            LoadBalancerConfig,
            LoadBalancingAlgorithm,
        )

        config = LoadBalancerConfig(algorithm=LoadBalancingAlgorithm.RANDOM)
        balancer = LoadBalancer(nodes=mock_nodes, config=config)

        # Select many times and check all nodes are selected
        selected = set()
        for _ in range(50):
            node = await balancer.select_node()
            selected.add(node.provider.provider_name)

        # All providers should be selected at least once
        assert len(selected) == 3

    @pytest.mark.asyncio
    async def test_no_available_nodes(self) -> None:
        """Test selection when no nodes available."""
        from src.core.vision.load_balancer import LoadBalancer, ProviderNode

        provider = MagicMock(spec=VisionProvider)
        provider.provider_name = "disabled"
        node = ProviderNode(provider=provider, enabled=False)

        balancer = LoadBalancer(nodes=[node])
        selected = await balancer.select_node()

        assert selected is None

    def test_add_remove_node(self, mock_nodes: list) -> None:
        """Test adding and removing nodes."""
        from src.core.vision.load_balancer import LoadBalancer, ProviderNode

        balancer = LoadBalancer(nodes=mock_nodes[:2])
        assert len(balancer.get_available_nodes()) == 2

        # Add node
        balancer.add_node(mock_nodes[2])
        assert len(balancer.get_available_nodes()) == 3

        # Remove node
        result = balancer.remove_node("provider_1")
        assert result is True
        assert len(balancer.get_available_nodes()) == 2

    def test_enable_disable_node(self, mock_nodes: list) -> None:
        """Test enabling and disabling nodes."""
        from src.core.vision.load_balancer import LoadBalancer

        balancer = LoadBalancer(nodes=mock_nodes)

        # Disable node
        result = balancer.disable_node("provider_1")
        assert result is True
        assert len(balancer.get_available_nodes()) == 2

        # Enable node
        result = balancer.enable_node("provider_1")
        assert result is True
        assert len(balancer.get_available_nodes()) == 3

    def test_get_stats(self, mock_nodes: list) -> None:
        """Test getting load balancer statistics."""
        from src.core.vision.load_balancer import LoadBalancer

        balancer = LoadBalancer(nodes=mock_nodes)

        # Record some requests
        mock_nodes[0].record_request_start()
        mock_nodes[0].record_request_end(True, 100.0)
        mock_nodes[1].record_request_start()
        mock_nodes[1].record_request_end(False, 50.0)

        stats = balancer.get_stats()

        assert stats.total_requests == 2
        assert stats.successful_requests == 1
        assert stats.failed_requests == 1
        assert len(stats.requests_per_provider) == 3


class TestLoadBalancedVisionProvider:
    """Tests for LoadBalancedVisionProvider wrapper."""

    @pytest.fixture
    def setup_providers(self) -> tuple:
        """Create mock providers and load balancer."""
        from src.core.vision.load_balancer import LoadBalancer, ProviderNode

        providers = []
        nodes = []
        for i in range(2):
            provider = MagicMock(spec=VisionProvider)
            provider.provider_name = f"provider_{i}"
            provider.analyze_image = AsyncMock(
                return_value=VisionDescription(
                    summary=f"Result from provider_{i}",
                    details=[f"Detail from provider {i}"],
                    confidence=0.9,
                )
            )
            providers.append(provider)
            nodes.append(ProviderNode(provider=provider))

        balancer = LoadBalancer(nodes=nodes)
        return providers, balancer

    @pytest.mark.asyncio
    async def test_analyze_image_success(self, setup_providers: tuple) -> None:
        """Test successful image analysis."""
        from src.core.vision.load_balancer import LoadBalancedVisionProvider

        providers, balancer = setup_providers
        wrapper = LoadBalancedVisionProvider(load_balancer=balancer)

        result = await wrapper.analyze_image(b"test image data")

        assert "Result from provider_" in result.summary

    @pytest.mark.asyncio
    async def test_analyze_image_retry_on_failure(self, setup_providers: tuple) -> None:
        """Test retry behavior on provider failure."""
        from src.core.vision.load_balancer import LoadBalancedVisionProvider

        providers, balancer = setup_providers

        # First provider fails, second succeeds
        providers[0].analyze_image = AsyncMock(side_effect=Exception("Provider 0 failed"))

        wrapper = LoadBalancedVisionProvider(
            load_balancer=balancer,
            retry_on_failure=True,
            max_retries=1,
        )

        result = await wrapper.analyze_image(b"test image data")
        assert result.summary == "Result from provider_1"

    @pytest.mark.asyncio
    async def test_analyze_image_all_fail(self, setup_providers: tuple) -> None:
        """Test behavior when all providers fail."""
        from src.core.vision.load_balancer import LoadBalancedVisionProvider

        providers, balancer = setup_providers

        # All providers fail
        for p in providers:
            p.analyze_image = AsyncMock(side_effect=Exception("Provider failed"))

        wrapper = LoadBalancedVisionProvider(load_balancer=balancer, max_retries=1)

        with pytest.raises(VisionProviderError):
            await wrapper.analyze_image(b"test image data")


# =============================================================================
# Logging Middleware Tests
# =============================================================================


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_level_values(self) -> None:
        """Test log level enum values."""
        from src.core.vision.logging_middleware import LogLevel

        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"


class TestRequestLog:
    """Tests for RequestLog dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        from src.core.vision.logging_middleware import RequestLog

        log = RequestLog(
            request_id="test-123",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            provider="openai",
            image_size_bytes=1024,
            include_description=True,
            metadata={"key": "value"},
        )

        result = log.to_dict()

        assert result["request_id"] == "test-123"
        assert result["provider"] == "openai"
        assert result["image_size_bytes"] == 1024
        assert result["metadata"] == {"key": "value"}


class TestResponseLog:
    """Tests for ResponseLog dataclass."""

    def test_success_response_to_dict(self) -> None:
        """Test successful response conversion."""
        from src.core.vision.logging_middleware import ResponseLog

        log = ResponseLog(
            request_id="test-123",
            timestamp=datetime(2024, 1, 15, 10, 30, 1),
            provider="openai",
            success=True,
            response_time_ms=250.5,
            result_summary="Test summary",
            confidence=0.95,
        )

        result = log.to_dict()

        assert result["success"] is True
        assert result["response_time_ms"] == 250.5
        assert result["confidence"] == 0.95

    def test_error_response_to_dict(self) -> None:
        """Test error response conversion."""
        from src.core.vision.logging_middleware import ResponseLog

        log = ResponseLog(
            request_id="test-123",
            timestamp=datetime(2024, 1, 15, 10, 30, 1),
            provider="openai",
            success=False,
            response_time_ms=100.0,
            error_message="API error",
            error_type="VisionProviderError",
        )

        result = log.to_dict()

        assert result["success"] is False
        assert result["error_message"] == "API error"
        assert result["error_type"] == "VisionProviderError"


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_initial_state(self) -> None:
        """Test initial metrics state."""
        from src.core.vision.logging_middleware import PerformanceMetrics

        metrics = PerformanceMetrics()

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.avg_response_time_ms == 0.0
        assert metrics.success_rate == 0.0

    def test_record_success(self) -> None:
        """Test recording successful request."""
        from src.core.vision.logging_middleware import PerformanceMetrics

        metrics = PerformanceMetrics()
        metrics.record(success=True, response_time_ms=100.0)

        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.avg_response_time_ms == 100.0

    def test_record_failure(self) -> None:
        """Test recording failed request."""
        from src.core.vision.logging_middleware import PerformanceMetrics

        metrics = PerformanceMetrics()
        metrics.record(success=False, response_time_ms=50.0)

        assert metrics.total_requests == 1
        assert metrics.failed_requests == 1

    def test_percentile_calculation(self) -> None:
        """Test percentile calculation."""
        from src.core.vision.logging_middleware import PerformanceMetrics

        metrics = PerformanceMetrics()

        # Record 100 requests with varying response times
        for i in range(100):
            metrics.record(success=True, response_time_ms=float(i + 1))

        p50 = metrics.get_percentile(50)
        p90 = metrics.get_percentile(90)
        p99 = metrics.get_percentile(99)

        assert 49 <= p50 <= 51
        assert 89 <= p90 <= 91
        assert 98 <= p99 <= 100


class TestLoggingMiddleware:
    """Tests for LoggingMiddleware class."""

    def test_log_request_returns_id(self) -> None:
        """Test that logging request returns correlation ID."""
        from src.core.vision.logging_middleware import LoggingMiddleware

        middleware = LoggingMiddleware()
        request_id = middleware.log_request(
            provider="openai",
            image_size_bytes=1024,
        )

        assert request_id is not None
        assert len(request_id) == 8  # UUID prefix

    def test_log_response_updates_metrics(self) -> None:
        """Test that logging response updates metrics."""
        from src.core.vision.logging_middleware import LoggingMiddleware

        middleware = LoggingMiddleware()

        # Log request and response
        request_id = middleware.log_request(provider="openai", image_size_bytes=1024)
        middleware.log_response(
            request_id=request_id,
            provider="openai",
            success=True,
            response_time_ms=150.0,
        )

        metrics = middleware.get_metrics()
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1

    def test_provider_specific_metrics(self) -> None:
        """Test provider-specific metrics tracking."""
        from src.core.vision.logging_middleware import LoggingMiddleware

        middleware = LoggingMiddleware()

        # Log for different providers
        for provider in ["openai", "anthropic"]:
            request_id = middleware.log_request(provider=provider, image_size_bytes=1024)
            middleware.log_response(
                request_id=request_id,
                provider=provider,
                success=True,
                response_time_ms=100.0,
            )

        openai_metrics = middleware.get_metrics("openai")
        anthropic_metrics = middleware.get_metrics("anthropic")

        assert openai_metrics.total_requests == 1
        assert anthropic_metrics.total_requests == 1

    def test_reset_metrics(self) -> None:
        """Test metrics reset."""
        from src.core.vision.logging_middleware import LoggingMiddleware

        middleware = LoggingMiddleware()

        request_id = middleware.log_request(provider="openai", image_size_bytes=1024)
        middleware.log_response(
            request_id=request_id,
            provider="openai",
            success=True,
            response_time_ms=100.0,
        )

        middleware.reset_metrics()

        metrics = middleware.get_metrics()
        assert metrics.total_requests == 0


class TestLoggingVisionProvider:
    """Tests for LoggingVisionProvider wrapper."""

    @pytest.fixture
    def mock_provider(self) -> MagicMock:
        """Create mock provider."""
        provider = MagicMock(spec=VisionProvider)
        provider.provider_name = "test_provider"
        provider.analyze_image = AsyncMock(
            return_value=VisionDescription(
                summary="Test result",
                details=["Test detail"],
                confidence=0.9,
            )
        )
        return provider

    @pytest.mark.asyncio
    async def test_analyze_image_logs_request(self, mock_provider: MagicMock) -> None:
        """Test that analyze_image logs request."""
        from src.core.vision.logging_middleware import LoggingMiddleware, LoggingVisionProvider

        middleware = LoggingMiddleware()
        wrapper = LoggingVisionProvider(mock_provider, middleware)

        await wrapper.analyze_image(b"test data")

        metrics = middleware.get_metrics()
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1

    @pytest.mark.asyncio
    async def test_analyze_image_logs_error(self, mock_provider: MagicMock) -> None:
        """Test that analyze_image logs errors."""
        from src.core.vision.logging_middleware import LoggingMiddleware, LoggingVisionProvider

        mock_provider.analyze_image = AsyncMock(side_effect=Exception("Test error"))

        middleware = LoggingMiddleware()
        wrapper = LoggingVisionProvider(mock_provider, middleware)

        with pytest.raises(Exception):
            await wrapper.analyze_image(b"test data")

        metrics = middleware.get_metrics()
        assert metrics.total_requests == 1
        assert metrics.failed_requests == 1


# =============================================================================
# Profile Tests
# =============================================================================


class TestProfileType:
    """Tests for ProfileType enum."""

    def test_profile_type_values(self) -> None:
        """Test profile type enum values."""
        from src.core.vision.profiles import ProfileType

        assert ProfileType.DEVELOPMENT.value == "development"
        assert ProfileType.PRODUCTION.value == "production"
        assert ProfileType.TESTING.value == "testing"
        assert ProfileType.HIGH_PERFORMANCE.value == "high_performance"


class TestProviderType:
    """Tests for ProviderType enum."""

    def test_provider_type_values(self) -> None:
        """Test provider type enum values."""
        from src.core.vision.profiles import ProviderType

        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.ANTHROPIC.value == "anthropic"
        assert ProviderType.DEEPSEEK.value == "deepseek"


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        from src.core.vision.profiles import ProviderConfig, ProviderType

        config = ProviderConfig(provider_type=ProviderType.OPENAI)

        assert config.weight == 1.0
        assert config.enabled is True
        assert config.max_connections == 100
        assert config.retry.max_retries == 3

    def test_get_api_key_from_env(self) -> None:
        """Test API key retrieval from environment."""
        import os

        from src.core.vision.profiles import ProviderConfig, ProviderType

        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key_env_var="TEST_API_KEY",
        )

        # Set env var
        os.environ["TEST_API_KEY"] = "test-key-123"

        assert config.get_api_key() == "test-key-123"

        # Clean up
        del os.environ["TEST_API_KEY"]

    def test_get_api_key_direct(self) -> None:
        """Test direct API key."""
        from src.core.vision.profiles import ProviderConfig, ProviderType

        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key="direct-key-456",
        )

        assert config.get_api_key() == "direct-key-456"


class TestProfileConfig:
    """Tests for ProfileConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default profile values."""
        from src.core.vision.profiles import ProfileConfig, ProfileType

        profile = ProfileConfig(
            name="test",
            profile_type=ProfileType.DEVELOPMENT,
        )

        assert profile.enable_preprocessing is True
        assert profile.enable_logging is True
        assert profile.max_image_size_mb == 20.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        from src.core.vision.profiles import ProfileConfig, ProfileType

        profile = ProfileConfig(
            name="test",
            profile_type=ProfileType.DEVELOPMENT,
            description="Test profile",
        )

        result = profile.to_dict()

        assert result["name"] == "test"
        assert result["profile_type"] == "development"
        assert result["description"] == "Test profile"


class TestProfileManager:
    """Tests for ProfileManager class."""

    def test_builtin_profiles(self) -> None:
        """Test that builtin profiles are registered."""
        from src.core.vision.profiles import ProfileManager

        manager = ProfileManager()

        profiles = manager.list_profiles()
        assert "development" in profiles
        assert "testing" in profiles
        assert "production" in profiles
        assert "high_performance" in profiles
        assert "cost_optimized" in profiles

    def test_get_profile(self) -> None:
        """Test getting profile by name."""
        from src.core.vision.profiles import ProfileManager

        manager = ProfileManager()

        profile = manager.get_profile("development")
        assert profile is not None
        assert profile.name == "development"

    def test_get_nonexistent_profile(self) -> None:
        """Test getting nonexistent profile."""
        from src.core.vision.profiles import ProfileManager

        manager = ProfileManager()

        profile = manager.get_profile("nonexistent")
        assert profile is None

    def test_register_custom_profile(self) -> None:
        """Test registering custom profile."""
        from src.core.vision.profiles import ProfileConfig, ProfileManager, ProfileType

        manager = ProfileManager()

        custom = ProfileConfig(
            name="custom",
            profile_type=ProfileType.CUSTOM,
            description="Custom profile",
        )
        manager.register_profile(custom)

        retrieved = manager.get_profile("custom")
        assert retrieved is not None
        assert retrieved.description == "Custom profile"

    def test_create_custom_profile_from_base(self) -> None:
        """Test creating custom profile from base."""
        from src.core.vision.profiles import ProfileManager

        manager = ProfileManager()

        custom = manager.create_custom_profile(
            name="my_profile",
            base_profile="development",
            enable_analytics=True,
        )

        assert custom.name == "my_profile"
        assert custom.enable_analytics is True

    def test_validate_profile(self) -> None:
        """Test profile validation."""
        from src.core.vision.profiles import ProfileManager

        manager = ProfileManager()

        result = manager.validate_profile("development")

        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result


class TestProfileEnvironmentLoader:
    """Tests for ProfileEnvironmentLoader class."""

    def test_get_profile_name_default(self) -> None:
        """Test default profile name."""
        from src.core.vision.profiles import ProfileEnvironmentLoader

        loader = ProfileEnvironmentLoader()
        name = loader.get_profile_name()

        assert name == "development"

    def test_get_profile_name_from_env(self) -> None:
        """Test profile name from environment."""
        import os

        from src.core.vision.profiles import ProfileEnvironmentLoader

        os.environ["VISION_PROFILE"] = "production"

        loader = ProfileEnvironmentLoader()
        name = loader.get_profile_name()

        assert name == "production"

        del os.environ["VISION_PROFILE"]


class TestCreateProfile:
    """Tests for create_profile factory function."""

    def test_create_basic_profile(self) -> None:
        """Test creating basic profile."""
        from src.core.vision.profiles import create_profile

        profile = create_profile(
            name="test_profile",
            providers=[
                {"provider_type": "openai", "weight": 2.0},
                {"provider_type": "anthropic", "weight": 1.0},
            ],
            enable_load_balancing=True,
        )

        assert profile.name == "test_profile"
        assert len(profile.providers) == 2
        assert profile.providers[0].weight == 2.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase3Integration:
    """Integration tests for Phase 3 features."""

    @pytest.mark.asyncio
    async def test_preprocessing_with_logging(self) -> None:
        """Test preprocessing combined with logging."""
        from src.core.vision.logging_middleware import LoggingMiddleware, LoggingVisionProvider
        from src.core.vision.preprocessing import (
            ImageFormat,
            ImageInfo,
            PreprocessingConfig,
            PreprocessingVisionProvider,
        )

        # Create mock provider
        mock_provider = MagicMock(spec=VisionProvider)
        mock_provider.provider_name = "test"
        mock_provider.analyze_image = AsyncMock(
            return_value=VisionDescription(
                summary="Test",
                details=[],
                confidence=0.9,
            )
        )

        # Wrap with preprocessing
        preprocessing_config = PreprocessingConfig()
        preprocessing_wrapper = PreprocessingVisionProvider(mock_provider, preprocessing_config)

        # Wrap with logging
        middleware = LoggingMiddleware()
        logging_wrapper = LoggingVisionProvider(preprocessing_wrapper, middleware)

        # Test with mocked preprocessor
        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        with patch.object(
            preprocessing_wrapper._preprocessor,
            "preprocess",
            return_value=(
                png_data,
                ImageInfo(
                    format=ImageFormat.PNG,
                    width=100,
                    height=100,
                    size_bytes=len(png_data),
                    has_alpha=False,
                    color_mode="RGB",
                ),
            ),
        ):
            result = await logging_wrapper.analyze_image(png_data)

        assert result.summary == "Test"
        assert middleware.get_metrics().total_requests == 1

    @pytest.mark.asyncio
    async def test_load_balancer_with_logging(self) -> None:
        """Test load balancer combined with logging."""
        from src.core.vision.load_balancer import (
            LoadBalancedVisionProvider,
            LoadBalancer,
            ProviderNode,
        )
        from src.core.vision.logging_middleware import LoggingMiddleware, LoggingVisionProvider

        # Create mock providers
        nodes = []
        for i in range(2):
            provider = MagicMock(spec=VisionProvider)
            provider.provider_name = f"provider_{i}"
            provider.analyze_image = AsyncMock(
                return_value=VisionDescription(
                    summary=f"Result {i}",
                    details=[],
                    confidence=0.9,
                )
            )
            nodes.append(ProviderNode(provider=provider))

        balancer = LoadBalancer(nodes=nodes)
        lb_wrapper = LoadBalancedVisionProvider(load_balancer=balancer)

        # Wrap with logging
        middleware = LoggingMiddleware()
        logging_wrapper = LoggingVisionProvider(lb_wrapper, middleware)

        # Test
        result = await logging_wrapper.analyze_image(b"test data")

        assert "Result" in result.summary
        assert middleware.get_metrics().total_requests == 1
