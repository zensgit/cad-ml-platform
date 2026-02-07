from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.core.providers import (
    BaseProvider,
    ProviderConfig,
    ProviderRegistry,
    ProviderStatus,
)


@dataclass
class DemoConfig(ProviderConfig):
    token: str = "demo"


class DemoProvider(BaseProvider[DemoConfig, dict]):
    async def _process_impl(self, request, **kwargs):
        return {"request": request, "provider": self.name}


class BrokenHealthProvider(BaseProvider[DemoConfig, dict]):
    async def _process_impl(self, request, **kwargs):
        return {"ok": True}

    async def _health_check_impl(self) -> bool:
        raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def _clear_registry():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


def test_registry_register_and_list():
    ProviderRegistry.register("vision", "demo")(DemoProvider)
    assert ProviderRegistry.list_domains() == ["vision"]
    assert ProviderRegistry.list_providers("vision") == ["demo"]
    assert ProviderRegistry.exists("vision", "demo") is True


def test_registry_duplicate_registration_rejected():
    ProviderRegistry.register("vision", "demo")(DemoProvider)
    with pytest.raises(ValueError, match="already registered"):
        ProviderRegistry.register("vision", "demo")(DemoProvider)


def test_registry_get_unknown_raises():
    with pytest.raises(KeyError, match="Provider not found"):
        ProviderRegistry.get_provider_class("vision", "missing")


@pytest.mark.asyncio
async def test_registry_get_creates_provider_and_process():
    ProviderRegistry.register("vision", "demo")(DemoProvider)
    provider = ProviderRegistry.get(
        "vision",
        "demo",
        DemoConfig(name="demo_provider", provider_type="vision"),
    )
    result = await provider.process({"hello": "world"})
    assert result["provider"] == "demo_provider"
    assert result["request"] == {"hello": "world"}


@pytest.mark.asyncio
async def test_health_check_updates_status_and_snapshot():
    provider = DemoProvider(DemoConfig(name="demo", provider_type="vision"))
    assert provider.status == ProviderStatus.UNKNOWN

    ok = await provider.health_check()
    assert ok is True
    assert provider.status == ProviderStatus.HEALTHY
    snapshot = provider.status_snapshot()
    assert snapshot["name"] == "demo"
    assert snapshot["provider_type"] == "vision"
    assert snapshot["status"] == "healthy"
    assert snapshot["last_health_check_at"] is not None
    assert snapshot["last_health_check_latency_ms"] is not None


@pytest.mark.asyncio
async def test_health_check_failure_is_captured():
    provider = BrokenHealthProvider(DemoConfig(name="broken", provider_type="vision"))
    ok = await provider.health_check()
    assert ok is False
    assert provider.status == ProviderStatus.DOWN
    assert provider.last_error == "boom"


def test_unregister_removes_provider():
    ProviderRegistry.register("vision", "demo")(DemoProvider)
    removed = ProviderRegistry.unregister("vision", "demo")
    assert removed is True
    assert ProviderRegistry.exists("vision", "demo") is False
    assert ProviderRegistry.list_domains() == []
