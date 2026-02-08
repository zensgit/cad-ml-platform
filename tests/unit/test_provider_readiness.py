from __future__ import annotations

import pytest

from src.core.providers.base import BaseProvider, ProviderConfig
from src.core.providers.readiness import check_provider_readiness
from src.core.providers.registry import ProviderRegistry


class _DummyProvider(BaseProvider[ProviderConfig, dict]):
    async def _health_check_impl(self) -> bool:
        return bool(self.config.metadata.get("ok", True))

    async def _process_impl(self, request, **kwargs):  # type: ignore[no-untyped-def]
        return {"status": "ok"}


@pytest.fixture(autouse=True)
def _clear_registry():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


def _register(domain: str, name: str, ok: bool) -> None:
    @ProviderRegistry.register(domain, name)
    class _P(_DummyProvider):  # noqa: D401
        def __init__(self, config: ProviderConfig | None = None):
            super().__init__(
                config
                or ProviderConfig(
                    name=name,
                    provider_type=domain,
                    metadata={"ok": ok},
                )
            )


@pytest.mark.asyncio
async def test_check_provider_readiness_required_and_optional() -> None:
    _register("test", "ok", ok=True)
    _register("test", "down", ok=False)

    summary = await check_provider_readiness(
        required=[("test", "ok")],
        optional=[("test", "down")],
        timeout_seconds=0.2,
    )

    assert summary.ok is True
    assert summary.degraded is True
    assert "test/ok" in summary.required
    assert "test/down" in summary.optional
    assert summary.required_down == []
    assert summary.optional_down == ["test/down"]


@pytest.mark.asyncio
async def test_check_provider_readiness_fails_when_required_down() -> None:
    _register("test", "down", ok=False)

    summary = await check_provider_readiness(
        required=[("test", "down")],
        optional=[],
        timeout_seconds=0.2,
    )

    assert summary.ok is False
    assert summary.degraded is False
    assert summary.required_down == ["test/down"]
