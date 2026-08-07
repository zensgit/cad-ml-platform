"""§8.3 fail-closed external AI cost cap + isolated sample mode."""

from __future__ import annotations

import os

import pytest

from src.core.assistant.cost_cap import (
    ENV_EXTERNAL_AI_COST_CAP_USD,
    ENV_EXTERNAL_AI_SPEND_USD,
    ENV_ISOLATED_SAMPLE_MODE,
    CostCapRejected,
    assert_external_ai_allowed,
)
from src.core.assistant.llm_providers import get_provider


@pytest.fixture(autouse=True)
def _clear_cost_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        ENV_EXTERNAL_AI_COST_CAP_USD,
        ENV_EXTERNAL_AI_SPEND_USD,
        ENV_ISOLATED_SAMPLE_MODE,
        "ASSISTANT_HOSTED_PROVIDER_OPT_IN",
    ):
        monkeypatch.delenv(key, raising=False)


def test_isolated_sample_mode_blocks_external_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_ISOLATED_SAMPLE_MODE, "true")
    monkeypatch.setenv("ASSISTANT_HOSTED_PROVIDER_OPT_IN", "true")
    monkeypatch.setenv(ENV_EXTERNAL_AI_COST_CAP_USD, "100")
    with pytest.raises(CostCapRejected, match="ISOLATED_SAMPLE_MODE"):
        assert_external_ai_allowed(provider_name="openai")


def test_hosted_requires_positive_cost_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSISTANT_HOSTED_PROVIDER_OPT_IN", "true")
    with pytest.raises(CostCapRejected, match="EXTERNAL_AI_COST_CAP_USD"):
        assert_external_ai_allowed(provider_name="openai")


def test_spend_at_cap_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSISTANT_HOSTED_PROVIDER_OPT_IN", "true")
    monkeypatch.setenv(ENV_EXTERNAL_AI_COST_CAP_USD, "10")
    monkeypatch.setenv(ENV_EXTERNAL_AI_SPEND_USD, "10")
    with pytest.raises(CostCapRejected, match="at or above cost cap"):
        assert_external_ai_allowed(provider_name="openai")


def test_spend_under_cap_allows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSISTANT_HOSTED_PROVIDER_OPT_IN", "true")
    monkeypatch.setenv(ENV_EXTERNAL_AI_COST_CAP_USD, "10")
    monkeypatch.setenv(ENV_EXTERNAL_AI_SPEND_USD, "1.5")
    assert_external_ai_allowed(provider_name="openai")  # no raise


def test_get_provider_hosted_without_cap_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real entry: get_provider('openai') with opt-in must hit cost-cap fail-closed."""
    monkeypatch.setenv("ASSISTANT_HOSTED_PROVIDER_OPT_IN", "true")
    # Cap missing → refuse construction of hosted provider.
    with pytest.raises(CostCapRejected):
        get_provider("openai")


def test_get_provider_offline_does_not_require_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline path remains available without a cost cap (no external AI)."""
    # No opt-in, no cap.
    provider = get_provider("offline")
    assert provider is not None
    assert provider.__class__.__name__ == "OfflineProvider"


def test_get_provider_hosted_opt_in_with_cap_constructs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASSISTANT_HOSTED_PROVIDER_OPT_IN", "true")
    monkeypatch.setenv(ENV_EXTERNAL_AI_COST_CAP_USD, "50")
    monkeypatch.setenv(ENV_EXTERNAL_AI_SPEND_USD, "0")
    # May still be unavailable without API keys, but construction must not CostCapRejected.
    provider = get_provider("openai")
    assert provider.__class__.__name__ == "OpenAIProvider"
