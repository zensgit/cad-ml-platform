"""Additional tests for classifier provider adapters to improve coverage.

Targets uncovered code paths in src/core/providers/classifier.py
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.core.providers import ProviderRegistry
from src.core.providers.classifier import (
    ClassifierProviderConfig,
    ClassifierRequest,
    Graph2DClassifierProviderAdapter,
    HybridClassifierProviderAdapter,
    V6PartClassifierProviderAdapter,
    V16PartClassifierProviderAdapter,
    bootstrap_core_classifier_providers,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


# --- ClassifierRequest Tests ---


class TestClassifierRequest:
    """Tests for ClassifierRequest dataclass."""

    def test_request_with_all_fields(self):
        req = ClassifierRequest(
            filename="test.dxf",
            file_bytes=b"content",
            file_path="/path/to/test.dxf",
        )
        assert req.filename == "test.dxf"
        assert req.file_bytes == b"content"
        assert req.file_path == "/path/to/test.dxf"

    def test_request_with_minimal_fields(self):
        req = ClassifierRequest(filename="test.dxf")
        assert req.filename == "test.dxf"
        assert req.file_bytes is None
        assert req.file_path is None


# --- HybridClassifierProviderAdapter Tests ---


class TestHybridClassifierProviderAdapter:
    """Tests for HybridClassifierProviderAdapter."""

    @pytest.mark.asyncio
    async def test_process_rejects_non_request_type(self):
        config = ClassifierProviderConfig(
            name="hybrid", provider_type="classifier", provider_name="hybrid"
        )
        provider = HybridClassifierProviderAdapter(config)

        with pytest.raises(TypeError, match="expects ClassifierRequest"):
            await provider.process({"invalid": "type"})

    @pytest.mark.asyncio
    async def test_process_rejects_empty_filename(self):
        config = ClassifierProviderConfig(
            name="hybrid", provider_type="classifier", provider_name="hybrid"
        )
        provider = HybridClassifierProviderAdapter(config)

        with pytest.raises(ValueError, match="filename cannot be empty"):
            await provider.process(ClassifierRequest(filename=""))

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_exception(self):
        config = ClassifierProviderConfig(
            name="hybrid", provider_type="classifier", provider_name="hybrid"
        )
        mock_classifier = MagicMock()
        mock_classifier.classify.side_effect = Exception("Test error")

        provider = HybridClassifierProviderAdapter(config, wrapped_classifier=mock_classifier)
        ok = await provider.health_check()
        assert ok is False
        assert provider.status.value == "down"


# --- Graph2DClassifierProviderAdapter Tests ---


class TestGraph2DClassifierProviderAdapter:
    """Tests for Graph2DClassifierProviderAdapter."""

    @pytest.mark.asyncio
    async def test_process_rejects_empty_file_bytes(self):
        config = ClassifierProviderConfig(
            name="graph2d", provider_type="classifier", provider_name="graph2d"
        )
        mock_classifier = MagicMock()
        provider = Graph2DClassifierProviderAdapter(config, wrapped_classifier=mock_classifier)

        with pytest.raises(ValueError, match="file_bytes cannot be empty"):
            await provider.process(ClassifierRequest(filename="test.dxf", file_bytes=None))

    @pytest.mark.asyncio
    async def test_process_rejects_empty_filename(self):
        config = ClassifierProviderConfig(
            name="graph2d", provider_type="classifier", provider_name="graph2d"
        )
        mock_classifier = MagicMock()
        provider = Graph2DClassifierProviderAdapter(config, wrapped_classifier=mock_classifier)

        with pytest.raises(ValueError, match="filename cannot be empty"):
            await provider.process(ClassifierRequest(filename="", file_bytes=b"content"))

    @pytest.mark.asyncio
    async def test_process_adds_ensemble_enabled_to_dict_result(self):
        config = ClassifierProviderConfig(
            name="graph2d", provider_type="classifier", provider_name="graph2d"
        )
        mock_classifier = MagicMock()
        mock_classifier.predict_from_bytes.return_value = {"label": "test"}

        provider = Graph2DClassifierProviderAdapter(
            config, wrapped_classifier=mock_classifier, ensemble=True
        )
        result = await provider.process(
            ClassifierRequest(filename="test.dxf", file_bytes=b"content")
        )

        assert result["ensemble_enabled"] is True

    @pytest.mark.asyncio
    async def test_health_check_disabled_by_config(self):
        config = ClassifierProviderConfig(
            name="graph2d", provider_type="classifier", provider_name="graph2d"
        )
        mock_classifier = MagicMock()
        provider = Graph2DClassifierProviderAdapter(config, wrapped_classifier=mock_classifier)

        with patch.dict(os.environ, {"GRAPH2D_ENABLED": "false"}):
            ok = await provider.health_check()
            assert ok is False
            assert "disabled_by_config" in (provider.last_error or "")

    @pytest.mark.asyncio
    async def test_health_check_torch_missing(self):
        config = ClassifierProviderConfig(
            name="graph2d", provider_type="classifier", provider_name="graph2d"
        )
        mock_classifier = MagicMock()
        provider = Graph2DClassifierProviderAdapter(config, wrapped_classifier=mock_classifier)

        with patch.dict(os.environ, {"GRAPH2D_ENABLED": "true"}):
            with patch("importlib.util.find_spec", return_value=None):
                ok = await provider.health_check()
                assert ok is False
                assert "torch_missing" in (provider.last_error or "")

    @pytest.mark.asyncio
    async def test_health_check_ensemble_model_missing(self):
        config = ClassifierProviderConfig(
            name="graph2d_ensemble", provider_type="classifier", provider_name="graph2d_ensemble"
        )
        mock_classifier = MagicMock()
        provider = Graph2DClassifierProviderAdapter(
            config, wrapped_classifier=mock_classifier, ensemble=True
        )

        with patch.dict(os.environ, {"GRAPH2D_ENABLED": "true", "GRAPH2D_ENSEMBLE_MODELS": ""}):
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                with patch("os.path.exists", return_value=False):
                    ok = await provider.health_check()
                    assert ok is False
                    assert "model_missing" in (provider.last_error or "")

    @pytest.mark.asyncio
    async def test_health_check_single_model_missing(self):
        config = ClassifierProviderConfig(
            name="graph2d", provider_type="classifier", provider_name="graph2d"
        )
        mock_classifier = MagicMock()
        provider = Graph2DClassifierProviderAdapter(
            config, wrapped_classifier=mock_classifier, ensemble=False
        )

        with patch.dict(os.environ, {"GRAPH2D_ENABLED": "true"}):
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                with patch("os.path.exists", return_value=False):
                    ok = await provider.health_check()
                    assert ok is False
                    assert "model_missing" in (provider.last_error or "")


# --- V16PartClassifierProviderAdapter Tests ---


class TestV16PartClassifierProviderAdapter:
    """Tests for V16PartClassifierProviderAdapter."""

    @pytest.mark.asyncio
    async def test_process_rejects_non_request_type(self):
        config = ClassifierProviderConfig(
            name="v16", provider_type="classifier", provider_name="v16"
        )
        provider = V16PartClassifierProviderAdapter(config)

        with pytest.raises(TypeError, match="expects ClassifierRequest"):
            await provider.process({"invalid": "type"})

    @pytest.mark.asyncio
    async def test_process_rejects_empty_file_path(self):
        config = ClassifierProviderConfig(
            name="v16", provider_type="classifier", provider_name="v16"
        )
        provider = V16PartClassifierProviderAdapter(config)

        with pytest.raises(ValueError, match="file_path is required"):
            await provider.process(ClassifierRequest(filename="test.dxf", file_path=None))

    @pytest.mark.asyncio
    async def test_process_returns_unavailable_when_classifier_none(self):
        config = ClassifierProviderConfig(
            name="v16", provider_type="classifier", provider_name="v16"
        )
        provider = V16PartClassifierProviderAdapter(config)

        with patch("src.core.analyzer._get_v16_classifier", return_value=None):
            result = await provider.process(
                ClassifierRequest(filename="test.dxf", file_path="/path/to/test.dxf")
            )
            assert result["status"] == "unavailable"

    @pytest.mark.asyncio
    async def test_health_check_disabled_by_config(self):
        config = ClassifierProviderConfig(
            name="v16", provider_type="classifier", provider_name="v16"
        )
        provider = V16PartClassifierProviderAdapter(config)

        with patch.dict(os.environ, {"DISABLE_V16_CLASSIFIER": "true"}):
            ok = await provider.health_check()
            assert ok is False
            assert "disabled_by_config" in (provider.last_error or "")

    @pytest.mark.asyncio
    async def test_health_check_torch_missing(self):
        config = ClassifierProviderConfig(
            name="v16", provider_type="classifier", provider_name="v16"
        )
        provider = V16PartClassifierProviderAdapter(config)

        with patch.dict(os.environ, {"DISABLE_V16_CLASSIFIER": ""}):
            with patch.object(V16PartClassifierProviderAdapter, "_has_torch", return_value=False):
                ok = await provider.health_check()
                assert ok is False
                assert "torch_missing" in (provider.last_error or "")

    @pytest.mark.asyncio
    async def test_health_check_models_missing(self):
        config = ClassifierProviderConfig(
            name="v16", provider_type="classifier", provider_name="v16"
        )
        provider = V16PartClassifierProviderAdapter(config)

        with patch.dict(os.environ, {"DISABLE_V16_CLASSIFIER": ""}):
            with patch.object(V16PartClassifierProviderAdapter, "_has_torch", return_value=True):
                with patch.object(
                    V16PartClassifierProviderAdapter, "_models_present", return_value=False
                ):
                    ok = await provider.health_check()
                    assert ok is False
                    assert "model_missing" in (provider.last_error or "")


# --- V6PartClassifierProviderAdapter Tests ---


class TestV6PartClassifierProviderAdapter:
    """Tests for V6PartClassifierProviderAdapter."""

    @pytest.mark.asyncio
    async def test_process_rejects_non_request_type(self):
        config = ClassifierProviderConfig(
            name="v6", provider_type="classifier", provider_name="v6"
        )
        provider = V6PartClassifierProviderAdapter(config)

        with pytest.raises(TypeError, match="expects ClassifierRequest"):
            await provider.process({"invalid": "type"})

    @pytest.mark.asyncio
    async def test_process_rejects_empty_file_path(self):
        config = ClassifierProviderConfig(
            name="v6", provider_type="classifier", provider_name="v6"
        )
        provider = V6PartClassifierProviderAdapter(config)

        with pytest.raises(ValueError, match="file_path is required"):
            await provider.process(ClassifierRequest(filename="test.dxf", file_path=None))

    @pytest.mark.asyncio
    async def test_health_check_torch_missing(self):
        config = ClassifierProviderConfig(
            name="v6", provider_type="classifier", provider_name="v6"
        )
        provider = V6PartClassifierProviderAdapter(config)

        with patch.object(V6PartClassifierProviderAdapter, "_has_torch", return_value=False):
            ok = await provider.health_check()
            assert ok is False
            assert "torch_missing" in (provider.last_error or "")

    @pytest.mark.asyncio
    async def test_health_check_model_missing(self):
        config = ClassifierProviderConfig(
            name="v6", provider_type="classifier", provider_name="v6"
        )
        provider = V6PartClassifierProviderAdapter(config)

        with patch.object(V6PartClassifierProviderAdapter, "_has_torch", return_value=True):
            with patch.object(V6PartClassifierProviderAdapter, "_model_present", return_value=False):
                ok = await provider.health_check()
                assert ok is False
                assert "model_missing" in (provider.last_error or "")


# --- Bootstrap Tests ---


class TestBootstrapIdempotent:
    """Test that bootstrap is idempotent."""

    def test_bootstrap_is_idempotent(self):
        bootstrap_core_classifier_providers()
        first_hybrid = ProviderRegistry.get_provider_class("classifier", "hybrid")

        # Call again - should not raise
        bootstrap_core_classifier_providers()
        second_hybrid = ProviderRegistry.get_provider_class("classifier", "hybrid")

        # Should be the same class
        assert first_hybrid is second_hybrid
