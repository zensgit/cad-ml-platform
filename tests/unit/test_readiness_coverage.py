"""Additional tests for readiness module to improve coverage.

Targets uncovered code paths in src/core/providers/readiness.py:
- Lines 34-48: parse_provider_id_list various parsing branches
- Line 78: to_dict() method
- Line 109: timeout <= 0 default handling
- Lines 122-123: provider init exception handling
- Line 151: duplicate provider ID deduplication
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.core.providers.base import BaseProvider, ProviderConfig
from src.core.providers.readiness import (
    ProviderReadinessItem,
    ProviderReadinessSummary,
    check_provider_readiness,
    format_provider_id,
    load_provider_readiness_config_from_env,
    parse_provider_id_list,
)
from src.core.providers.registry import ProviderRegistry


# --- Fixtures ---


@pytest.fixture(autouse=True)
def _clear_registry():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


class DummyProvider(BaseProvider[ProviderConfig, dict]):
    """Dummy provider for testing."""

    async def _health_check_impl(self) -> bool:
        return bool(self.config.metadata.get("ok", True))

    async def _process_impl(self, request, **kwargs):
        return {"status": "ok"}


def _register(domain: str, name: str, ok: bool = True) -> None:
    """Helper to register a dummy provider."""

    @ProviderRegistry.register(domain, name)
    class _P(DummyProvider):
        def __init__(self, config: ProviderConfig | None = None):
            super().__init__(
                config
                or ProviderConfig(
                    name=name,
                    provider_type=domain,
                    metadata={"ok": ok},
                )
            )


# --- parse_provider_id_list Tests (Lines 34-48) ---


class TestParseProviderIdList:
    """Tests for parse_provider_id_list function."""

    def test_empty_string_returns_empty_list(self):
        """Line 32-33: Empty string returns empty list."""
        assert parse_provider_id_list("") == []
        assert parse_provider_id_list("   ") == []

    def test_parse_slash_separator(self):
        """Line 39-40: Parse 'domain/name' format."""
        result = parse_provider_id_list("classifier/hybrid")
        assert result == [("classifier", "hybrid")]

    def test_parse_colon_separator(self):
        """Line 41-42: Parse 'domain:name' format."""
        result = parse_provider_id_list("ocr:paddle")
        assert result == [("ocr", "paddle")]

    def test_parse_multiple_providers_comma_separated(self):
        """Line 34: Parse comma-separated list."""
        result = parse_provider_id_list("classifier/hybrid, ocr/paddle, vision/openai")
        assert result == [
            ("classifier", "hybrid"),
            ("ocr", "paddle"),
            ("vision", "openai"),
        ]

    def test_parse_multiple_providers_space_separated(self):
        """Line 34: Parse space-separated list."""
        result = parse_provider_id_list("classifier/hybrid ocr/paddle")
        assert result == [("classifier", "hybrid"), ("ocr", "paddle")]

    def test_parse_mixed_separators(self):
        """Parse mixed comma and space separators."""
        result = parse_provider_id_list("classifier/hybrid, ocr:paddle vision/openai")
        assert result == [
            ("classifier", "hybrid"),
            ("ocr", "paddle"),
            ("vision", "openai"),
        ]

    def test_invalid_token_without_separator_ignored(self):
        """Lines 45-46: Invalid tokens without separator are ignored."""
        result = parse_provider_id_list("invalid classifier/hybrid also_invalid")
        assert result == [("classifier", "hybrid")]

    def test_empty_domain_ignored(self):
        """Lines 45-46: Empty domain is ignored."""
        result = parse_provider_id_list("/hybrid classifier/valid")
        assert result == [("classifier", "valid")]

    def test_empty_name_ignored(self):
        """Lines 45-46: Empty name is ignored."""
        result = parse_provider_id_list("classifier/ ocr/paddle")
        assert result == [("ocr", "paddle")]

    def test_whitespace_in_tokens_trimmed(self):
        """Lines 43-44: Whitespace around domain and name is trimmed."""
        # Note: space in "classifier / hybrid" splits into separate tokens
        # Only "classifier/hybrid" (no spaces around /) is valid
        result = parse_provider_id_list("  classifier/hybrid  ,  ocr/paddle  ")
        assert result == [("classifier", "hybrid"), ("ocr", "paddle")]


# --- format_provider_id Tests ---


class TestFormatProviderId:
    """Tests for format_provider_id function."""

    def test_format_basic(self):
        """Format provider ID tuple to string."""
        assert format_provider_id(("classifier", "hybrid")) == "classifier/hybrid"

    def test_format_various(self):
        """Format various provider IDs."""
        assert format_provider_id(("ocr", "paddle")) == "ocr/paddle"
        assert format_provider_id(("vision", "openai")) == "vision/openai"


# --- ProviderReadinessItem Tests ---


class TestProviderReadinessItem:
    """Tests for ProviderReadinessItem dataclass."""

    def test_item_with_all_fields(self):
        """Create item with all fields."""
        item = ProviderReadinessItem(
            id="classifier/hybrid",
            ready=True,
            error=None,
            checked_at=1234567890.0,
            latency_ms=5.5,
        )
        assert item.id == "classifier/hybrid"
        assert item.ready is True
        assert item.error is None
        assert item.checked_at == 1234567890.0
        assert item.latency_ms == 5.5

    def test_item_with_error(self):
        """Create item with error."""
        item = ProviderReadinessItem(
            id="classifier/hybrid",
            ready=False,
            error="timeout",
            checked_at=1234567890.0,
            latency_ms=500.0,
        )
        assert item.ready is False
        assert item.error == "timeout"


# --- ProviderReadinessSummary.to_dict Tests (Line 78) ---


class TestProviderReadinessSummaryToDict:
    """Tests for ProviderReadinessSummary.to_dict method."""

    def test_to_dict_basic(self):
        """Line 78: to_dict returns proper dict structure."""
        item = ProviderReadinessItem(
            id="classifier/hybrid",
            ready=True,
            error=None,
            checked_at=1234567890.0,
            latency_ms=5.5,
        )
        summary = ProviderReadinessSummary(
            ok=True,
            degraded=False,
            required=["classifier/hybrid"],
            optional=["ocr/paddle"],
            required_down=[],
            optional_down=["ocr/paddle"],
            timeout_seconds=0.5,
            checked_at=1234567890.0,
            results=[item],
        )

        result = summary.to_dict()

        assert result["ok"] is True
        assert result["degraded"] is False
        assert result["required"] == ["classifier/hybrid"]
        assert result["optional"] == ["ocr/paddle"]
        assert result["required_down"] == []
        assert result["optional_down"] == ["ocr/paddle"]
        assert result["timeout_seconds"] == 0.5
        assert result["checked_at"] == 1234567890.0
        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "classifier/hybrid"
        assert result["results"][0]["ready"] is True

    def test_to_dict_with_multiple_results(self):
        """to_dict with multiple results."""
        items = [
            ProviderReadinessItem(id="a/b", ready=True, error=None),
            ProviderReadinessItem(id="c/d", ready=False, error="down"),
        ]
        summary = ProviderReadinessSummary(
            ok=True,
            degraded=True,
            required=["a/b"],
            optional=["c/d"],
            required_down=[],
            optional_down=["c/d"],
            timeout_seconds=1.0,
            checked_at=0.0,
            results=items,
        )

        result = summary.to_dict()
        assert len(result["results"]) == 2
        assert result["results"][1]["error"] == "down"


# --- check_provider_readiness Edge Cases ---


class TestCheckProviderReadinessEdgeCases:
    """Tests for check_provider_readiness edge cases."""

    @pytest.mark.asyncio
    async def test_zero_timeout_uses_default(self):
        """Line 108-109: timeout <= 0 uses 0.5 as default."""
        _register("test", "ok", ok=True)

        summary = await check_provider_readiness(
            required=[("test", "ok")],
            optional=[],
            timeout_seconds=0,
        )

        assert summary.ok is True
        assert summary.timeout_seconds == 0.5

    @pytest.mark.asyncio
    async def test_negative_timeout_uses_default(self):
        """Line 108-109: negative timeout uses 0.5 as default."""
        _register("test", "ok", ok=True)

        summary = await check_provider_readiness(
            required=[("test", "ok")],
            optional=[],
            timeout_seconds=-5,
        )

        assert summary.ok is True
        assert summary.timeout_seconds == 0.5

    @pytest.mark.asyncio
    async def test_large_timeout_capped_at_10(self):
        """Line 110: timeout capped at 10 seconds."""
        _register("test", "ok", ok=True)

        summary = await check_provider_readiness(
            required=[("test", "ok")],
            optional=[],
            timeout_seconds=100,
        )

        assert summary.ok is True
        assert summary.timeout_seconds == 10.0

    @pytest.mark.asyncio
    async def test_provider_init_exception_handled(self):
        """Lines 122-123: Exception during provider init is captured."""

        # Register a provider that raises on instantiation
        @ProviderRegistry.register("test", "broken")
        class BrokenProvider(DummyProvider):
            def __init__(self, config=None):
                raise RuntimeError("Init failed")

        summary = await check_provider_readiness(
            required=[("test", "broken")],
            optional=[],
            timeout_seconds=0.5,
        )

        assert summary.ok is False
        assert "test/broken" in summary.required_down
        # Check error message contains init_error
        for item in summary.results:
            if item.id == "test/broken":
                assert "init_error" in (item.error or "")
                break

    @pytest.mark.asyncio
    async def test_duplicate_provider_deduplication(self):
        """Line 150-153: Duplicate providers are deduplicated."""
        _register("test", "ok", ok=True)

        summary = await check_provider_readiness(
            required=[("test", "ok"), ("test", "ok")],  # Duplicate
            optional=[("test", "ok")],  # Also duplicate
            timeout_seconds=0.5,
        )

        assert summary.ok is True
        # Should only have one result for test/ok
        assert len(summary.results) == 1
        assert summary.results[0].id == "test/ok"

    @pytest.mark.asyncio
    async def test_empty_provider_lists(self):
        """Empty required and optional lists."""
        summary = await check_provider_readiness(
            required=[],
            optional=[],
            timeout_seconds=0.5,
        )

        assert summary.ok is True
        assert summary.degraded is False
        assert summary.results == []

    @pytest.mark.asyncio
    async def test_only_optional_providers(self):
        """Only optional providers (no required)."""
        _register("test", "down", ok=False)

        summary = await check_provider_readiness(
            required=[],
            optional=[("test", "down")],
            timeout_seconds=0.5,
        )

        assert summary.ok is True  # No required providers = ok
        assert summary.degraded is True  # But degraded because optional is down


# --- load_provider_readiness_config_from_env Tests ---


class TestLoadProviderReadinessConfigFromEnv:
    """Tests for load_provider_readiness_config_from_env function."""

    def test_empty_env_vars(self):
        """Empty environment variables return empty lists."""
        with patch.dict(
            os.environ,
            {"READINESS_REQUIRED_PROVIDERS": "", "READINESS_OPTIONAL_PROVIDERS": ""},
            clear=False,
        ):
            required, optional = load_provider_readiness_config_from_env()
            assert required == []
            assert optional == []

    def test_load_required_providers(self):
        """Load required providers from env."""
        with patch.dict(
            os.environ,
            {
                "READINESS_REQUIRED_PROVIDERS": "classifier/hybrid, ocr/paddle",
                "READINESS_OPTIONAL_PROVIDERS": "",
            },
            clear=False,
        ):
            required, optional = load_provider_readiness_config_from_env()
            assert required == [("classifier", "hybrid"), ("ocr", "paddle")]
            assert optional == []

    def test_load_optional_providers(self):
        """Load optional providers from env."""
        with patch.dict(
            os.environ,
            {
                "READINESS_REQUIRED_PROVIDERS": "",
                "READINESS_OPTIONAL_PROVIDERS": "vision/openai",
            },
            clear=False,
        ):
            required, optional = load_provider_readiness_config_from_env()
            assert required == []
            assert optional == [("vision", "openai")]

    def test_load_both_required_and_optional(self):
        """Load both required and optional from env."""
        with patch.dict(
            os.environ,
            {
                "READINESS_REQUIRED_PROVIDERS": "classifier/hybrid",
                "READINESS_OPTIONAL_PROVIDERS": "ocr:paddle vision/openai",
            },
            clear=False,
        ):
            required, optional = load_provider_readiness_config_from_env()
            assert required == [("classifier", "hybrid")]
            assert optional == [("ocr", "paddle"), ("vision", "openai")]

    def test_whitespace_handling(self):
        """Whitespace around env values is trimmed."""
        with patch.dict(
            os.environ,
            {
                "READINESS_REQUIRED_PROVIDERS": "  classifier/hybrid  ",
                "READINESS_OPTIONAL_PROVIDERS": "  ocr/paddle  ",
            },
            clear=False,
        ):
            required, optional = load_provider_readiness_config_from_env()
            assert required == [("classifier", "hybrid")]
            assert optional == [("ocr", "paddle")]
