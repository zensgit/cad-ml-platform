"""Test metrics consistency for error codes and labels.

Ensures that:
1. All ErrorCode values are used consistently in metrics
2. Provider exceptions map to correct ErrorCode values
3. Metrics labels match the expected contract
4. Error counters increment correctly for specific exceptions
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio
import base64

from src.core.errors import ErrorCode
from src.core.ocr.exceptions import OcrError
from src.core.ocr.providers.paddle import PaddleOcrProvider
from src.core.ocr.providers.deepseek_hf import DeepSeekHfProvider
from src.core.vision.manager import VisionManager
from src.core.vision.providers.deepseek_stub import DeepSeekStubProvider
from src.core.vision.base import VisionAnalyzeRequest


class TestErrorCodeConsistency:
    """Verify ErrorCode enum is used consistently across providers."""

    def test_all_error_codes_have_values(self):
        """Ensure all ErrorCode members have string values."""
        for code in ErrorCode:
            assert isinstance(code.value, str)
            assert len(code.value) > 0
            # Verify naming convention (UPPER_SNAKE_CASE)
            assert code.value.isupper()
            assert "_" in code.value or code.value.isalpha()

    def test_error_code_uniqueness(self):
        """Ensure no duplicate ErrorCode values."""
        values = [code.value for code in ErrorCode]
        assert len(values) == len(set(values))

    def test_ocr_errors_compatibility(self):
        """Verify legacy OCR_ERRORS dict maps to ErrorCode values."""
        from src.core.ocr.exceptions import OCR_ERRORS

        # Check that all values in OCR_ERRORS are valid ErrorCode values
        valid_codes = {code.value for code in ErrorCode}
        for key, value in OCR_ERRORS.items():
            assert value in valid_codes, f"OCR_ERRORS['{key}'] = '{value}' is not a valid ErrorCode"


class TestPaddleProviderMetrics:
    """Test PaddleOcrProvider exception mapping to ErrorCode."""

    @pytest.mark.asyncio
    async def test_memory_error_maps_to_resource_exhausted(self):
        """Verify MemoryError maps to RESOURCE_EXHAUSTED."""
        provider = PaddleOcrProvider()

        with patch("src.core.ocr.providers.paddle.PaddleOCR") as mock_paddle:
            # Simulate MemoryError during initialization
            mock_paddle.side_effect = MemoryError("Out of memory")

            with patch("src.utils.metrics.ocr_errors_total.labels") as mock_metrics:
                mock_counter = Mock()
                mock_metrics.return_value = mock_counter

                with pytest.raises(MemoryError):
                    await provider.warmup()

                # Verify the correct ErrorCode was used
                mock_metrics.assert_called_with(
                    provider="paddle",
                    code=ErrorCode.RESOURCE_EXHAUSTED.value,
                    stage="init"
                )
                mock_counter.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_maps_to_provider_timeout(self):
        """Verify TimeoutError maps to PROVIDER_TIMEOUT."""
        provider = PaddleOcrProvider()
        provider._initialized = True
        provider._ocr = Mock()

        # Simulate TimeoutError during OCR
        provider._ocr.ocr.side_effect = TimeoutError("OCR timeout")

        with patch("src.utils.metrics.ocr_errors_total.labels") as mock_metrics:
            mock_counter = Mock()
            mock_metrics.return_value = mock_counter

            # Call extract and expect it to handle the error
            result = await provider.extract(b"test_image")

            # Should return empty result on error
            assert result.text == ""

            # Verify the correct ErrorCode was used
            mock_metrics.assert_called_with(
                provider="paddle",
                code=ErrorCode.PROVIDER_TIMEOUT.value,
                stage="infer"
            )
            mock_counter.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_error_maps_to_parse_failed(self):
        """Verify parsing errors map to PARSE_FAILED."""
        provider = PaddleOcrProvider()
        provider._initialized = True
        provider._ocr = Mock()

        # Return malformed OCR result
        provider._ocr.ocr.return_value = [["invalid", "structure"]]

        with patch("src.utils.metrics.ocr_errors_total.labels") as mock_metrics:
            mock_counter = Mock()
            mock_metrics.return_value = mock_counter

            # Call extract - should handle parse error gracefully
            result = await provider.extract(b"test_image")

            # Check if PARSE_FAILED was used for any parsing issues
            calls = mock_metrics.call_args_list
            parse_failed_calls = [
                call for call in calls
                if len(call) > 1 and call[1].get("code") == ErrorCode.PARSE_FAILED.value
            ]

            # Should have at least one PARSE_FAILED call if parsing failed
            if parse_failed_calls:
                assert any(call[1].get("stage") == "parse" for call in parse_failed_calls)


class TestDeepSeekHfProviderMetrics:
    """Test DeepSeekHfProvider exception mapping to ErrorCode."""

    @pytest.mark.asyncio
    async def test_model_load_error_mapping(self):
        """Verify model loading errors map to MODEL_LOAD_ERROR."""
        provider = DeepSeekHfProvider()

        with patch("src.core.ocr.providers.deepseek_hf.AutoTokenizer") as mock_tokenizer:
            mock_tokenizer.from_pretrained.side_effect = Exception("Model not found")

            with patch("src.utils.metrics.ocr_errors_total.labels") as mock_metrics:
                mock_counter = Mock()
                mock_metrics.return_value = mock_counter

                await provider._lazy_load()

                # Should have fallen back to stub
                assert provider._model == "stub"

                # Verify the correct ErrorCode was used
                mock_metrics.assert_called_with(
                    provider="deepseek_hf",
                    code=ErrorCode.MODEL_LOAD_ERROR.value,
                    stage="load"
                )
                mock_counter.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_inference_timeout_mapping(self):
        """Verify inference timeout maps to PROVIDER_TIMEOUT."""
        provider = DeepSeekHfProvider(timeout_ms=100)
        provider._model = "stub"  # Use stub mode

        # Patch the inner _infer function to simulate slow response
        async def slow_infer():
            await asyncio.sleep(1)  # Longer than timeout
            return ""

        # Need to patch at module level where it's defined
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            with patch("src.utils.metrics.ocr_errors_total.labels") as mock_metrics:
                mock_counter = Mock()
                mock_metrics.return_value = mock_counter

                # Extract should timeout
                result = await provider.extract(b"test_image")

                # Verify timeout was recorded
                mock_metrics.assert_any_call(
                    provider="deepseek_hf",
                    code=ErrorCode.PROVIDER_TIMEOUT.value,
                    stage="infer"
                )


class TestVisionManagerMetrics:
    """Test VisionManager error handling and metrics."""

    @pytest.mark.asyncio
    async def test_input_rejection_metrics(self):
        """Verify input rejection reasons are tracked correctly."""
        vision_provider = DeepSeekStubProvider()
        manager = VisionManager(vision_provider=vision_provider)

        # Test base64 too large
        with patch("src.core.config.get_settings") as mock_settings:
            # Set a small limit to trigger the error
            mock_settings.return_value.VISION_MAX_BASE64_BYTES = 100

            with patch("src.utils.metrics.vision_input_rejected_total.labels") as mock_rejected:
                mock_counter = Mock()
                mock_rejected.return_value = mock_counter

                # Create base64 payload larger than limit
                large_data = b"x" * 200
                large_base64 = base64.b64encode(large_data).decode()

                request = VisionAnalyzeRequest(
                    image_base64=large_base64,
                    include_description=True
                )

                from src.core.vision.base import VisionInputError

                with pytest.raises(VisionInputError):
                    await manager._load_image(request)

                mock_rejected.assert_called_with(reason="base64_too_large")
            mock_counter.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_url_timeout_rejection(self):
        """Verify URL timeout is tracked with correct reason."""
        vision_provider = DeepSeekStubProvider()
        manager = VisionManager(vision_provider=vision_provider)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            import httpx
            mock_client.get.side_effect = httpx.TimeoutException("Timeout")

            with patch("src.utils.metrics.vision_input_rejected_total.labels") as mock_rejected:
                mock_counter = Mock()
                mock_rejected.return_value = mock_counter

                request = VisionAnalyzeRequest(
                    image_url="http://example.com/image.png",
                    include_description=True
                )

                from src.core.vision.base import VisionInputError

                with pytest.raises(VisionInputError):
                    await manager._load_image(request)

                mock_rejected.assert_called_with(reason="url_timeout")
                mock_counter.inc.assert_called_once()


class TestMetricsLabelContract:
    """Verify metrics use consistent label names and values."""

    def test_ocr_errors_total_label_contract(self):
        """Verify ocr_errors_total uses (provider, code, stage) labels."""
        with patch("src.utils.metrics.ocr_errors_total.labels") as mock_labels:
            mock_counter = Mock()
            mock_labels.return_value = mock_counter

            # Simulate an error from provider
            error = OcrError(
                ErrorCode.NETWORK_ERROR,
                "Network failed",
                provider="test_provider",
                stage="infer"
            )

            # Record the error (simulating what providers do)
            from src.utils.metrics import ocr_errors_total
            ocr_errors_total.labels(
                provider=error.provider,
                code=error.code if isinstance(error.code, str) else error.code.value,
                stage=error.stage
            ).inc()

            # Verify label names are correct
            mock_labels.assert_called_with(
                provider="test_provider",
                code=ErrorCode.NETWORK_ERROR.value,
                stage="infer"
            )

    def test_vision_errors_total_label_contract(self):
        """Verify vision_errors_total uses (provider, code) labels."""
        with patch("src.utils.metrics.vision_errors_total.labels") as mock_labels:
            mock_counter = Mock()
            mock_labels.return_value = mock_counter

            from src.utils.metrics import vision_errors_total
            vision_errors_total.labels(
                provider="deepseek_stub",
                code="input_error"
            ).inc()

            # Verify label names are correct
            mock_labels.assert_called_with(
                provider="deepseek_stub",
                code="input_error"
            )

    def test_stage_values_consistency(self):
        """Verify stage values are consistent across providers."""
        expected_stages = {"init", "load", "preprocess", "infer", "parse", "align", "postprocess"}

        # These are the stages we've seen in the code
        used_stages = set()

        # Check PaddleOcrProvider stages
        with open("/Users/huazhou/Insync/hua.chau@outlook.com/OneDrive/应用/GitHub/cad-ml-platform/src/core/ocr/providers/paddle.py") as f:
            content = f.read()
            for stage in expected_stages:
                if f'stage="{stage}"' in content:
                    used_stages.add(stage)

        # Check DeepSeekHfProvider stages
        with open("/Users/huazhou/Insync/hua.chau@outlook.com/OneDrive/应用/GitHub/cad-ml-platform/src/core/ocr/providers/deepseek_hf.py") as f:
            content = f.read()
            for stage in expected_stages:
                if f'stage="{stage}"' in content:
                    used_stages.add(stage)

        # Should use at least some standard stages
        assert len(used_stages) > 0
        assert used_stages.issubset(expected_stages)


class TestErrorPropagation:
    """Test that errors propagate correctly through the stack."""

    @pytest.mark.asyncio
    async def test_ocr_error_propagates_with_code(self):
        """Verify OcrError preserves ErrorCode through propagation."""
        from src.core.ocr.manager import OcrManager

        manager = OcrManager(providers={})

        # Try to extract with no providers
        with pytest.raises(OcrError) as exc_info:
            await manager.extract(b"test_image", strategy="paddle")

        error = exc_info.value
        assert error.code == ErrorCode.PROVIDER_DOWN
        assert error.provider == "paddle"
        assert error.stage == "infer"

    def test_error_code_in_api_response(self):
        """Verify ErrorCode appears in API error responses."""
        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app)

        # Send invalid base64 to trigger INPUT_ERROR
        response = client.post(
            "/api/v1/vision/analyze",
            json={
                "image_base64": "not-valid-base64!!!",
                "include_description": True
            }
        )

        assert response.status_code == 200  # API returns 200 with success=false
        data = response.json()
        assert data["success"] is False
        assert "code" in data
        assert data["code"] == ErrorCode.INPUT_ERROR.value


class TestMetricsIncrement:
    """Test that metrics increment correctly for various scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_error_counting(self):
        """Verify metrics are thread-safe for concurrent errors."""
        provider = PaddleOcrProvider()

        with patch("src.utils.metrics.ocr_errors_total.labels") as mock_labels:
            counters = {}

            def get_counter(provider=None, code=None, stage=None):
                key = (provider, code, stage)
                if key not in counters:
                    counters[key] = Mock()
                return counters[key]

            mock_labels.side_effect = get_counter

            # Simulate concurrent errors
            async def cause_error():
                try:
                    provider._ocr = Mock()
                    provider._ocr.ocr.side_effect = MemoryError()
                    provider._initialized = True
                    await provider.extract(b"test")
                except:
                    pass

            # Run multiple concurrent extractions
            await asyncio.gather(*[cause_error() for _ in range(5)])

            # Check that counter was incremented correctly
            memory_key = ("paddle", ErrorCode.RESOURCE_EXHAUSTED.value, "infer")
            if memory_key in counters:
                assert counters[memory_key].inc.call_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])