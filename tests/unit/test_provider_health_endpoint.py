"""Tests for provider health check endpoint (/api/v1/providers/health)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.core.providers.base import BaseProvider, ProviderConfig
from src.main import app


@dataclass
class DemoConfig(ProviderConfig):
    pass


class DemoProvider(BaseProvider[DemoConfig, dict]):
    async def _process_impl(self, request, **kwargs):
        return {"ok": True}

    async def _health_check_impl(self) -> bool:
        return True


def test_provider_health_endpoint_returns_sorted_results():
    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}

    def _list_domains():
        return ["vision", "classifier"]

    def _list_providers(domain: str):
        if domain == "vision":
            return ["stub"]
        if domain == "classifier":
            return ["hybrid"]
        return []

    def _get(domain: str, provider_name: str):
        return DemoProvider(
            DemoConfig(name=f"{domain}/{provider_name}", provider_type=domain)
        )

    with patch("src.core.providers.bootstrap_core_provider_registry", return_value={}):
        with patch(
            "src.core.providers.get_core_provider_registry_snapshot",
            return_value={
                "plugins": {
                    "configured": ["tests.fixtures.provider_plugin_example:bootstrap"],
                    "loaded": ["tests.fixtures.provider_plugin_example:bootstrap"],
                    "errors": [],
                    "cache": {"reused": True, "reason": "config_match"},
                    "summary": {
                        "overall_status": "ok",
                        "configured_count": 1,
                        "loaded_count": 1,
                        "error_count": 0,
                    },
                }
            },
        ):
            with patch(
                "src.core.providers.ProviderRegistry.list_domains",
                side_effect=_list_domains,
            ):
                with patch(
                    "src.core.providers.ProviderRegistry.list_providers",
                    side_effect=_list_providers,
                ):
                    with patch("src.core.providers.ProviderRegistry.get", side_effect=_get):
                        resp = client.get(
                            "/api/v1/providers/health",
                            headers=headers,
                            params={"timeout_seconds": 0.1},
                        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert payload["total"] == 2
    assert payload["ready"] == 2
    assert payload["timeout_seconds"] == 0.1
    assert [r["domain"] for r in payload["results"]] == ["classifier", "vision"]
    assert [r["provider"] for r in payload["results"]] == ["hybrid", "stub"]
    diagnostics = payload.get("plugin_diagnostics") or {}
    assert diagnostics.get("configured_count") == 1
    assert diagnostics.get("loaded_count") == 1
    assert diagnostics.get("error_count") == 0
    assert diagnostics.get("summary", {}).get("overall_status") == "ok"
