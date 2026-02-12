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


def test_provider_health_endpoint_sanitizes_plugin_errors():
    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}

    long_error = "boom\nSECRET=abc\n" + ("x" * 1000)
    with patch("src.core.providers.bootstrap_core_provider_registry", return_value={}):
        with patch(
            "src.core.providers.get_core_provider_registry_snapshot",
            return_value={
                "plugins": {
                    "configured": ["tests.fixtures.provider_plugin_example:bootstrap"],
                    "loaded": [],
                    "errors": [{"plugin": "tests.fixtures.provider_plugin_example:bootstrap", "error": long_error}],
                    "cache": {"reused": False, "reason": "first_load"},
                    "summary": {
                        "overall_status": "degraded",
                        "configured_count": 1,
                        "loaded_count": 0,
                        "error_count": 1,
                    },
                }
            },
        ):
            with patch(
                "src.core.providers.ProviderRegistry.list_domains",
                return_value=[],
            ):
                resp = client.get(
                    "/api/v1/providers/health",
                    headers=headers,
                    params={"timeout_seconds": 0.1},
                )

    assert resp.status_code == 200
    payload = resp.json()
    diagnostics = payload.get("plugin_diagnostics") or {}
    assert diagnostics.get("error_count") == 1
    errors_sample = diagnostics.get("errors_sample") or []
    assert len(errors_sample) == 1
    err = errors_sample[0].get("error") or ""
    assert "\n" not in err
    assert len(err) <= 300


def test_provider_health_endpoint_sanitizes_provider_last_error_and_error_field():
    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}

    long_error = "boom\nSECRET=abc\n" + ("x" * 1000)

    class ErrorProvider:
        name = "stub"
        provider_type = "vision"
        last_error = RuntimeError(long_error)

        async def health_check(self, timeout_seconds: float = 0.1) -> bool:
            return False

    with patch("src.core.providers.bootstrap_core_provider_registry", return_value={}):
        with patch(
            "src.core.providers.get_core_provider_registry_snapshot",
            return_value={
                "plugins": {
                    "configured": [],
                    "loaded": [],
                    "errors": [],
                    "cache": {"reused": True, "reason": "test"},
                    "summary": {
                        "overall_status": "ok",
                        "configured_count": 0,
                        "loaded_count": 0,
                        "error_count": 0,
                    },
                }
            },
        ):
            with patch(
                "src.core.providers.ProviderRegistry.list_domains",
                return_value=["vision"],
            ):
                with patch(
                    "src.core.providers.ProviderRegistry.list_providers",
                    return_value=["stub"],
                ):
                    with patch(
                        "src.core.providers.ProviderRegistry.get",
                        return_value=ErrorProvider(),
                    ):
                        resp = client.get(
                            "/api/v1/providers/health",
                            headers=headers,
                            params={"timeout_seconds": 0.1},
                        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload.get("total") == 1
    assert payload.get("ready") == 0

    result = payload["results"][0]
    assert result["domain"] == "vision"
    assert result["provider"] == "stub"
    assert result["ready"] is False

    err = result.get("error") or ""
    assert "\n" not in err
    assert len(err) <= 300

    snapshot = result.get("snapshot") or {}
    last_error = snapshot.get("last_error") or ""
    assert "\n" not in last_error
    assert len(last_error) <= 300


def test_provider_registry_endpoint_sanitizes_plugin_errors():
    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}

    long_error = "boom\nSECRET=abc\n" + ("x" * 2000)
    snapshot = {
        "bootstrapped": True,
        "bootstrap_timestamp": 0.0,
        "total_domains": 0,
        "total_providers": 0,
        "domains": ["vision", "ocr"],
        "providers": {"vision": [], "ocr": []},
        "provider_classes": {},
        "plugins": {
            "enabled": True,
            "strict": False,
            "configured": ["tests.fixtures.provider_plugin_example:bootstrap"],
            "loaded": [],
            "errors": [
                {
                    "plugin": "tests.fixtures.provider_plugin_example:bootstrap",
                    "error": long_error,
                }
            ],
            "registered": {},
            "cache": {"reused": False, "reason": "first_load"},
            "summary": {
                "overall_status": "degraded",
                "configured_count": 1,
                "loaded_count": 0,
                "error_count": 1,
            },
        },
    }

    with patch(
        "src.core.providers.get_core_provider_registry_snapshot",
        return_value=snapshot,
    ):
        resp = client.get(
            "/api/v1/providers/registry",
            headers=headers,
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload.get("status") == "ok"
    registry = payload.get("registry") or {}
    plugins = registry.get("plugins") or {}
    errors = plugins.get("errors") or []
    assert len(errors) == 1
    err = errors[0].get("error") or ""
    assert "\n" not in err
    assert len(err) <= 300


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
