from __future__ import annotations

import pytest

from src.api.health_utils import build_health_payload
from src.api.v1.health import hybrid_runtime_config, provider_registry_health


def test_health_payload_includes_ml_config_section() -> None:
    payload = build_health_payload()
    assert "config" in payload
    assert "ml" in payload["config"]
    assert "classification" in payload["config"]["ml"]
    assert "sampling" in payload["config"]["ml"]
    assert "hybrid_enabled" in payload["config"]["ml"]["classification"]
    assert "max_nodes" in payload["config"]["ml"]["sampling"]
    assert "core_providers" in payload["config"]
    assert "domains" in payload["config"]["core_providers"]
    assert "providers" in payload["config"]["core_providers"]


@pytest.mark.asyncio
async def test_health_hybrid_runtime_endpoint_returns_effective_config() -> None:
    payload = (await hybrid_runtime_config(api_key="test")).model_dump()
    assert payload["status"] == "ok"
    assert "config" in payload
    assert "filename" in payload["config"]
    assert "graph2d" in payload["config"]


@pytest.mark.asyncio
async def test_health_provider_registry_endpoint_returns_snapshot() -> None:
    payload = (await provider_registry_health(api_key="test")).model_dump()
    assert payload["status"] == "ok"
    assert "registry" in payload
    assert "domains" in payload["registry"]
    assert "providers" in payload["registry"]
    assert "vision" in payload["registry"]["domains"]
    assert "ocr" in payload["registry"]["domains"]
