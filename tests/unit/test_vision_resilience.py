"""Unit tests for vision provider resilience patterns.

Tests retry logic, circuit breaker, and metrics.
"""

import asyncio

import pytest

from src.core.vision import VisionDescription, VisionProviderError
from src.core.vision.resilience import (
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    ProviderMetrics,
    ResilientVisionProvider,
    RetryConfig,
    create_resilient_provider,
)

# Sample image data
SAMPLE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
)


class MockVisionProvider:
    """Mock provider for testing."""

    def __init__(self, fail_count: int = 0, latency: float = 0.01):
        self.fail_count = fail_count
        self.call_count = 0
        self.latency = latency

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True
    ) -> VisionDescription:
        self.call_count += 1
        await asyncio.sleep(self.latency)

        if self.call_count <= self.fail_count:
            raise VisionProviderError("mock", f"Simulated failure {self.call_count}")

        return VisionDescription(
            summary="Test summary",
            details=["Detail 1"],
            confidence=0.9,
        )

    @property
    def provider_name(self) -> str:
        return "mock"


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_values(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.exponential_base == 2.0

    def test_custom_values(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=10.0,
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 10.0


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_values(self):
        """Test default circuit breaker configuration."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout == 60.0

    def test_custom_values(self):
        """Test custom circuit breaker configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=1,
            timeout=30.0,
        )
        assert config.failure_threshold == 3
        assert config.success_threshold == 1
        assert config.timeout == 30.0


class TestProviderMetrics:
    """Tests for ProviderMetrics."""

    def test_initial_values(self):
        """Test initial metrics values."""
        metrics = ProviderMetrics()
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.success_rate == 1.0
        assert metrics.average_latency_ms == 0.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = ProviderMetrics(
            total_requests=100,
            successful_requests=90,
            failed_requests=10,
        )
        assert metrics.success_rate == 0.9

    def test_average_latency_calculation(self):
        """Test average latency calculation."""
        metrics = ProviderMetrics(
            successful_requests=10,
            total_latency_ms=5000.0,
        )
        assert metrics.average_latency_ms == 500.0


class TestResilientVisionProvider:
    """Tests for ResilientVisionProvider."""

    @pytest.mark.asyncio
    async def test_successful_request(self):
        """Test successful request without retries."""
        mock = MockVisionProvider(fail_count=0)
        resilient = ResilientVisionProvider(
            mock,
            retry_config=RetryConfig(max_retries=3, base_delay=0.01),
        )

        result = await resilient.analyze_image(SAMPLE_PNG)

        assert result.summary == "Test summary"
        assert mock.call_count == 1
        assert resilient.metrics.successful_requests == 1
        assert resilient.metrics.failed_requests == 0

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry on transient failures."""
        mock = MockVisionProvider(fail_count=2)
        resilient = ResilientVisionProvider(
            mock,
            retry_config=RetryConfig(max_retries=3, base_delay=0.01),
        )

        result = await resilient.analyze_image(SAMPLE_PNG)

        assert result.summary == "Test summary"
        assert mock.call_count == 3  # 2 failures + 1 success
        assert resilient.metrics.successful_requests == 1

    @pytest.mark.asyncio
    async def test_exhausted_retries(self):
        """Test failure after exhausted retries."""
        mock = MockVisionProvider(fail_count=10)  # Always fail
        resilient = ResilientVisionProvider(
            mock,
            retry_config=RetryConfig(max_retries=2, base_delay=0.01),
        )

        with pytest.raises(VisionProviderError, match="Failed after 3 attempts"):
            await resilient.analyze_image(SAMPLE_PNG)

        assert mock.call_count == 3  # Initial + 2 retries
        assert resilient.metrics.failed_requests == 3

    @pytest.mark.asyncio
    async def test_circuit_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        mock = MockVisionProvider(fail_count=100)
        resilient = ResilientVisionProvider(
            mock,
            retry_config=RetryConfig(max_retries=0, base_delay=0.01),
            circuit_config=CircuitBreakerConfig(
                failure_threshold=3,
                timeout=60.0,
            ),
        )

        # Trigger failures to open circuit
        for _ in range(3):
            try:
                await resilient.analyze_image(SAMPLE_PNG)
            except VisionProviderError:
                pass

        assert resilient.circuit_state == CircuitState.OPEN
        assert resilient.metrics.circuit_opens == 1

    @pytest.mark.asyncio
    async def test_circuit_rejects_when_open(self):
        """Test circuit breaker rejects requests when open."""
        mock = MockVisionProvider(fail_count=100)
        resilient = ResilientVisionProvider(
            mock,
            retry_config=RetryConfig(max_retries=0, base_delay=0.01),
            circuit_config=CircuitBreakerConfig(
                failure_threshold=2,
                timeout=60.0,
            ),
        )

        # Open the circuit
        for _ in range(2):
            try:
                await resilient.analyze_image(SAMPLE_PNG)
            except VisionProviderError:
                pass

        # Should raise CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await resilient.analyze_image(SAMPLE_PNG)

    @pytest.mark.asyncio
    async def test_circuit_half_open_recovery(self):
        """Test circuit breaker recovery from half-open."""
        mock = MockVisionProvider(fail_count=2)  # Fail first 2, then succeed
        resilient = ResilientVisionProvider(
            mock,
            retry_config=RetryConfig(max_retries=0, base_delay=0.01),
            circuit_config=CircuitBreakerConfig(
                failure_threshold=2,
                success_threshold=1,
                timeout=0.01,  # Very short for testing
            ),
        )

        # Open the circuit
        for _ in range(2):
            try:
                await resilient.analyze_image(SAMPLE_PNG)
            except VisionProviderError:
                pass

        assert resilient.circuit_state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.02)

        # Should succeed and close circuit
        result = await resilient.analyze_image(SAMPLE_PNG)
        assert result.summary == "Test summary"
        assert resilient.circuit_state == CircuitState.CLOSED

    def test_provider_name(self):
        """Test provider name is preserved."""
        mock = MockVisionProvider()
        resilient = ResilientVisionProvider(mock)
        assert resilient.provider_name == "mock"

    def test_reset_circuit(self):
        """Test manual circuit reset."""
        mock = MockVisionProvider()
        resilient = ResilientVisionProvider(mock)

        # Simulate some state
        resilient._circuit_state.state = CircuitState.OPEN
        resilient._circuit_state.failure_count = 10

        resilient.reset_circuit()

        assert resilient.circuit_state == CircuitState.CLOSED
        assert resilient._circuit_state.failure_count == 0

    def test_reset_metrics(self):
        """Test metrics reset."""
        mock = MockVisionProvider()
        resilient = ResilientVisionProvider(mock)

        # Simulate some metrics
        resilient._metrics.total_requests = 100
        resilient._metrics.failed_requests = 10

        resilient.reset_metrics()

        assert resilient.metrics.total_requests == 0
        assert resilient.metrics.failed_requests == 0


class TestCreateResilientProvider:
    """Tests for create_resilient_provider factory."""

    def test_creates_resilient_wrapper(self):
        """Test factory creates ResilientVisionProvider."""
        mock = MockVisionProvider()
        resilient = create_resilient_provider(mock)

        assert isinstance(resilient, ResilientVisionProvider)
        assert resilient.provider_name == "mock"

    def test_custom_retry_config(self):
        """Test factory with custom retry config."""
        mock = MockVisionProvider()
        resilient = create_resilient_provider(mock, max_retries=5)

        assert resilient._retry_config.max_retries == 5

    def test_custom_circuit_config(self):
        """Test factory with custom circuit config."""
        mock = MockVisionProvider()
        resilient = create_resilient_provider(
            mock,
            circuit_failure_threshold=10,
            circuit_timeout=120.0,
        )

        assert resilient._circuit_config.failure_threshold == 10
        assert resilient._circuit_config.timeout == 120.0
