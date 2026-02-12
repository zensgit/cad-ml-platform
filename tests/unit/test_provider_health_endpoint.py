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
    assert diagnostics.get("errors_sample") == []
    assert diagnostics.get("errors_truncated") is False
    assert diagnostics.get("registered_count") == 0
    assert diagnostics.get("registered_sample") == []
    assert diagnostics.get("summary", {}).get("overall_status") == "ok"


def test_provider_health_endpoint_supports_legacy_health_check_signature():
    """Legacy providers without timeout keyword should still be supported."""
    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}

    class LegacyProvider:
        name = "legacy_provider"
        provider_type = "legacy"
        last_error = None

        async def health_check(self):
            return True

    with patch("src.core.providers.bootstrap_core_provider_registry", return_value={}):
        with patch(
            "src.core.providers.ProviderRegistry.list_domains",
            return_value=["legacy"],
        ):
            with patch(
                "src.core.providers.ProviderRegistry.list_providers",
                side_effect=lambda d: ["legacy_provider"] if d == "legacy" else [],
            ):
                with patch(
                    "src.core.providers.ProviderRegistry.get",
                    return_value=LegacyProvider(),
                ):
                    resp = client.get(
                        "/api/v1/providers/health",
                        headers=headers,
                        params={"timeout_seconds": 0.1},
                    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert payload["total"] == 1
    assert payload["ready"] == 1
    result = payload["results"][0]
    assert result["domain"] == "legacy"
    assert result["provider"] == "legacy_provider"
    assert result["ready"] is True
    assert result["error"] is None
    assert result["snapshot"]["name"] == "legacy_provider"
    assert result["snapshot"]["provider_type"] == "legacy"
    assert result["snapshot"]["status"] == "unknown"
