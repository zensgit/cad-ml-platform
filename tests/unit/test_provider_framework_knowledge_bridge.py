from __future__ import annotations

import pytest

from src.core.providers import ProviderRegistry
from src.core.providers.knowledge import bootstrap_core_knowledge_providers


@pytest.fixture(autouse=True)
def _clear_registry():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


@pytest.mark.asyncio
async def test_bootstrap_core_knowledge_providers_registers_and_is_healthy():
    bootstrap_core_knowledge_providers()
    assert ProviderRegistry.exists("knowledge", "tolerance") is True
    assert ProviderRegistry.exists("knowledge", "standards") is True
    assert ProviderRegistry.exists("knowledge", "design_standards") is True

    tolerance = ProviderRegistry.get("knowledge", "tolerance")
    ok = await tolerance.health_check()
    assert ok is True
    payload = await tolerance.process({})
    assert payload.get("status") == "ok"
    assert isinstance(payload.get("counts"), dict)

    standards = ProviderRegistry.get("knowledge", "standards")
    ok = await standards.health_check()
    assert ok is True
    payload = await standards.process({})
    assert payload.get("status") == "ok"
    assert payload.get("counts", {}).get("threads", 0) > 0

    design_standards = ProviderRegistry.get("knowledge", "design_standards")
    ok = await design_standards.health_check()
    assert ok is True
    payload = await design_standards.process({})
    assert payload.get("status") == "ok"
    assert payload.get("counts", {}).get("surface_finish_grades", 0) > 0
