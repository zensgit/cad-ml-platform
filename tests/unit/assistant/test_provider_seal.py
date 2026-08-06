"""L3 SEAL discriminators for assistant provider egress (#535 §2 / §6).

Each test is designed so that temporarily reverting the corresponding seal
logic makes the assertion fail (mutation-verifiable).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.core.assistant.egress_allowlist import EgressRejected, validate_hosted_payload
from src.core.assistant.llm_providers import (
    OfflineProvider,
    get_best_available_provider,
    get_provider,
)
from src.core.assistant.provider_seal import (
    ENV_HOSTED_OPT_IN,
    ENV_LOCAL_ENDPOINT_ALLOWLIST,
    clear_provider_attempts,
    endpoint_is_verified_local,
    get_provider_attempts,
    hosted_provider_opt_in,
    is_hosted_provider_name,
    resolve_provider_name,
    sealed_fallback_chain,
)
from src.core.assistant.tool_status import (
    CANONICAL_NON_OK,
    is_citable_tool_result,
)
from src.core.assistant.tools import TOOL_REGISTRY


@pytest.fixture(autouse=True)
def _clean_seal_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_HOSTED_OPT_IN, raising=False)
    monkeypatch.delenv(ENV_LOCAL_ENDPOINT_ALLOWLIST, raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("VLLM_ENDPOINT", "http://localhost:8100")
    clear_provider_attempts()
    yield
    clear_provider_attempts()


# --- §6.1 no opt-in → Offline on all three path entry points ---


def test_no_opt_in_resolve_hosted_to_offline() -> None:
    assert not hosted_provider_opt_in()
    for name in ("claude", "openai", "qwen", "anthropic", "gpt4"):
        assert resolve_provider_name(name) == "offline"
        provider = get_provider(name)
        assert isinstance(provider, OfflineProvider)


def test_function_calling_default_is_offline() -> None:
    from src.core.assistant.function_calling import FunctionCallingEngine

    eng = FunctionCallingEngine()  # no args
    assert eng._provider_name == "offline"
    assert eng._anthropic_client is None
    assert eng._openai_client is None


def test_function_calling_hosted_without_opt_in_is_offline() -> None:
    from src.core.assistant.function_calling import FunctionCallingEngine

    eng = FunctionCallingEngine(llm_provider="claude")
    assert eng._provider_name == "offline"


def test_auto_select_without_opt_in_never_returns_hosted() -> None:
    # Even if hosted SDKs were present, sealed order excludes them without opt-in.
    with patch(
        "src.core.assistant.llm_providers.ClaudeProvider.is_available",
        return_value=True,
    ), patch(
        "src.core.assistant.llm_providers.OpenAIProvider.is_available",
        return_value=True,
    ):
        provider = get_best_available_provider()
        assert type(provider).__name__ in {
            "VLLMProvider",
            "OllamaProvider",
            "OfflineProvider",
        }


# --- §6.2 zero hosted attempts on fallback ---


def test_fallback_chain_contains_no_hosted() -> None:
    chain = sealed_fallback_chain()
    assert all(not is_hosted_provider_name(n) for n in chain)
    assert "claude" not in chain and "openai" not in chain and "qwen" not in chain


def test_assistant_fallback_records_zero_hosted_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core.assistant.assistant import AssistantConfig, CADAssistant

    clear_provider_attempts()

    # Primary provider fails; fallback must not touch hosted names.
    class _Boom:
        def is_available(self) -> bool:
            return True

        def generate(self, system_prompt: str, user_prompt: str) -> str:
            from src.core.assistant.provider_seal import record_provider_attempt

            record_provider_attempt("claude")
            raise RuntimeError("primary boom")

    cfg = AssistantConfig(auto_select_provider=False, llm_provider=__import__(
        "src.core.assistant.assistant", fromlist=["LLMProvider"]
    ).LLMProvider.LOCAL)
    assistant = CADAssistant(config=cfg)
    assistant._llm_provider = _Boom()  # type: ignore[assignment]

    # Ensure get_provider for hosted would record if called
    real_get = __import__(
        "src.core.assistant.assistant", fromlist=["get_provider"]
    ).get_provider

    def tracking_get(name: str, config=None):
        from src.core.assistant.provider_seal import record_provider_attempt

        # record the *requested* name before seal rewrite for attempt detection
        record_provider_attempt(f"request:{name}")
        return real_get(name, config)

    monkeypatch.setattr(
        "src.core.assistant.assistant.get_provider", tracking_get
    )

    out = assistant._fallback_generate("sys", "user")
    assert isinstance(out, str)
    attempts = get_provider_attempts()
    hosted_hits = [
        a
        for a in attempts
        if a in {"claude", "openai", "qwen", "anthropic", "gpt", "gpt4"}
        or a.startswith("request:claude")
        or a.startswith("request:openai")
        or a.startswith("request:qwen")
    ]
    # Primary boom recorded claude once; fallback must add zero hosted requests.
    assert "request:claude" not in attempts
    assert "request:openai" not in attempts
    assert "request:qwen" not in attempts
    # The explicit primary record may exist; filter those out of "fallback sequence"
    assert not any(a.startswith("request:") and is_hosted_provider_name(a.split(":", 1)[1]) for a in attempts)


def test_multi_model_select_with_fallback_drops_hosted_without_opt_in() -> None:
    from src.core.assistant.multi_model import (
        ModelConfig,
        ModelProvider,
        ModelSelector,
        ModelHealth,
        ModelStatus,
    )

    sel = ModelSelector()
    for prov, prio in [
        (ModelProvider.CLAUDE, 1),
        (ModelProvider.OPENAI, 2),
        (ModelProvider.OLLAMA, 3),
        (ModelProvider.OFFLINE, 4),
    ]:
        sel.register_model(
            ModelConfig(provider=prov, model_name="x", priority=prio)
        )
        sel._health[prov] = ModelHealth(prov, ModelStatus.AVAILABLE)

    ordered = sel.select_with_fallback()
    providers = {m.provider for m in ordered}
    assert ModelProvider.CLAUDE not in providers
    assert ModelProvider.OPENAI not in providers
    assert ModelProvider.OLLAMA in providers or ModelProvider.OFFLINE in providers


# --- §6.3 non-loopback endpoint requires opt-in ---


def test_non_loopback_endpoint_not_verified_local() -> None:
    assert endpoint_is_verified_local("http://localhost:11434")
    assert endpoint_is_verified_local("http://127.0.0.1:8100")
    assert not endpoint_is_verified_local("http://evil.example.com:8100")
    assert not endpoint_is_verified_local("http://10.0.0.5:8100")


def test_non_loopback_ollama_resolves_offline_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://evil.example.com:11434")
    assert resolve_provider_name("ollama") == "offline"
    assert isinstance(get_provider("ollama"), OfflineProvider)


def test_non_loopback_allowed_with_opt_in_and_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_HOSTED_OPT_IN, "1")
    monkeypatch.setenv(ENV_LOCAL_ENDPOINT_ALLOWLIST, "llm.internal")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://llm.internal:11434")
    assert endpoint_is_verified_local("http://llm.internal:11434")
    assert resolve_provider_name("ollama") == "ollama"


# --- §6.4 hosted payload allowlist ---


def test_payload_rejects_raw_bytes_and_ocr_and_paths() -> None:
    with pytest.raises(EgressRejected):
        validate_hosted_payload({"ocr_text": "secret drawing text"})
    with pytest.raises(EgressRejected):
        validate_hosted_payload({"file_path": "/Users/x/secret.dxf"})
    with pytest.raises(EgressRejected):
        validate_hosted_payload(b"\x00\x01raw")
    with pytest.raises(EgressRejected):
        validate_hosted_payload({"score": 0.9, "drawing_number": "DWG-001"})


def test_payload_allows_explainability_fields() -> None:
    validate_hosted_payload(
        {"score": 0.91, "confidence": 0.8, "rejection_reason_category": "geom_mismatch"}
    )


# --- §6.5 every TOOL_REGISTRY tool failure has canonical status ---


@pytest.mark.asyncio
async def test_all_tools_exception_path_canonical_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Iterate TOOL_REGISTRY — every tool's failure path is canonical status."""
    assert len(TOOL_REGISTRY) >= 9

    real_import = __import__

    def bad_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Force tool try-bodies that import src.* to fail into except.
        if isinstance(name, str) and (
            name.startswith("src.")
            or name in {"numpy", "PIL", "torch"}
        ):
            raise ImportError("blocked-by-test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", bad_import)

    params: Dict[str, Any] = {
        "file_id": "f1",
        "query": "304钢",
        "question": "材料",
        "action": "classify",
        "version": "v3",
    }
    for name, tool in TOOL_REGISTRY.items():
        result = await tool.execute(params)
        assert isinstance(result, dict), name
        assert result.get("status") in CANONICAL_NON_OK, (name, result)
        assert "reason_code" in result, name
        assert not is_citable_tool_result(result), name
        if name == "estimate_cost":
            assert "total" not in result


# --- §6.6 system prompt does not suppress non-ok status ---


def test_system_prompt_does_not_suppress_tool_failure() -> None:
    from src.core.assistant.function_calling import _get_system_prompt

    prompt = _get_system_prompt()
    assert "不需要向用户暴露" not in prompt
    assert "failed" in prompt or "degraded" in prompt or "unavailable" in prompt


# --- §6.8 missing SDK → offline deterministically ---


def test_missing_sdk_claude_is_offline_instance() -> None:
    # With no anthropic installed (normal), get_provider('claude') seals to Offline
    # without opt-in; with opt-in, ClaudeProvider is constructed but is_available False.
    p = get_provider("claude")
    assert isinstance(p, OfflineProvider)


def test_opt_in_claude_without_sdk_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_HOSTED_OPT_IN, "1")
    p = get_provider("claude")
    # May be ClaudeProvider with no client
    if type(p).__name__ == "ClaudeProvider":
        assert p.is_available() is False
    else:
        assert isinstance(p, OfflineProvider)
