from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.core.providers.base import BaseProvider, ProviderConfig
from src.core.providers.registry import ProviderRegistry


@dataclass
class ExampleProviderConfig(ProviderConfig):
    provider_name: str = "example"


class ExampleProvider(BaseProvider[ExampleProviderConfig, Dict[str, Any]]):
    def __init__(self, config: ExampleProviderConfig):
        super().__init__(config)

    async def _health_check_impl(self) -> bool:
        return True

    async def _process_impl(self, request: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = request
        _ = kwargs
        return {"status": "ok", "provider": "example"}


def bootstrap() -> None:
    """Register an example provider for unit tests."""
    if ProviderRegistry.exists("test", "example"):
        return

    @ProviderRegistry.register("test", "example")
    class ExampleCoreProvider(ExampleProvider):
        def __init__(self, config: Optional[ExampleProviderConfig] = None):
            cfg = config or ExampleProviderConfig(
                name="example",
                provider_type="test",
                provider_name="example",
            )
            super().__init__(config=cfg)

