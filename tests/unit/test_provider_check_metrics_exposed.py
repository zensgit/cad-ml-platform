"""Tests for provider check Prometheus metrics exposure.

These metrics are emitted as part of the core provider framework health/readiness
checks and should appear in `/metrics` once at least one labeled observation is
recorded.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.core.providers.base import BaseProvider, ProviderConfig
from src.main import app


@dataclass
class _DemoConfig(ProviderConfig):
    pass


class _DemoProvider(BaseProvider[_DemoConfig, dict]):
    async def _process_impl(self, request, **kwargs):  # type: ignore[no-untyped-def]
        return {"ok": True}

    async def _health_check_impl(self) -> bool:
        return True


def test_provider_health_emits_core_provider_metrics(require_metrics_enabled, metrics_text):
    """Provider health endpoint should emit per-provider counter + histogram."""
    client = TestClient(app)
    headers = {"X-API-Key": os.getenv("API_KEY", "test-key")}

    domain = "metrics_domain"
    provider_name = "metrics_provider"

    def _get(_domain: str, _provider: str):
        return _DemoProvider(_DemoConfig(name=f"{_domain}/{_provider}", provider_type=_domain))

    with patch("src.core.providers.bootstrap_core_provider_registry", return_value={}):
        with patch("src.core.providers.ProviderRegistry.list_domains", return_value=[domain]):
            with patch(
                "src.core.providers.ProviderRegistry.list_providers",
                side_effect=lambda d: [provider_name] if d == domain else [],
            ):
                with patch("src.core.providers.ProviderRegistry.get", side_effect=_get):
                    resp = client.get(
                        "/api/v1/providers/health",
                        headers=headers,
                        params={"timeout_seconds": 0.1},
                    )

    assert resp.status_code == 200

    text = metrics_text(client)
    if text:
        counter_lines = [
            line
            for line in text.splitlines()
            if line.startswith("core_provider_checks_total{")
            and f'domain="{domain}"' in line
            and f'provider="{provider_name}"' in line
            and 'result="ready"' in line
            and 'source="providers_health"' in line
        ]
        assert counter_lines, "missing labeled core_provider_checks_total for providers_health"

        duration_lines = [
            line
            for line in text.splitlines()
            if line.startswith("core_provider_check_duration_seconds_bucket{")
            and f'domain="{domain}"' in line
            and f'provider="{provider_name}"' in line
            and 'source="providers_health"' in line
        ]
        assert duration_lines, "missing labeled core_provider_check_duration_seconds_bucket for providers_health"


def test_provider_readiness_emits_core_provider_metrics(require_metrics_enabled, metrics_text):
    """Readiness checks should emit per-provider counter + histogram."""
    from src.core.providers.readiness import check_provider_readiness

    domain = "metrics_missing_domain"
    provider_name = "metrics_missing_provider"

    asyncio.run(
        check_provider_readiness(
            required=[(domain, provider_name)],
            optional=[],
            timeout_seconds=0.1,
        )
    )

    text = metrics_text()
    if text:
        counter_lines = [
            line
            for line in text.splitlines()
            if line.startswith("core_provider_checks_total{")
            and f'domain="{domain}"' in line
            and f'provider="{provider_name}"' in line
            and 'result="init_error"' in line
            and 'source="readiness"' in line
        ]
        assert counter_lines, "missing labeled core_provider_checks_total for readiness init_error"

        duration_lines = [
            line
            for line in text.splitlines()
            if line.startswith("core_provider_check_duration_seconds_bucket{")
            and f'domain="{domain}"' in line
            and f'provider="{provider_name}"' in line
            and 'source="readiness"' in line
        ]
        assert duration_lines, "missing labeled core_provider_check_duration_seconds_bucket for readiness"
