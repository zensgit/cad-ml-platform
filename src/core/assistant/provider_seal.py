"""L3 provider SEAL — local-only defaults, opt-in hosted egress, endpoint proof.

Implements design-lock ``L3_ASSISTANT_SCOPE_SEAL_DESIGNLOCK_20260805.md`` §2.A
(and endpoint half of §2.A). Runtime behavior only; §3 REDESIGN is out of scope.

Env
---
``ASSISTANT_HOSTED_PROVIDER_OPT_IN``
    When truthy (``1``/``true``/``yes``/``on``), hosted providers (Claude/OpenAI/Qwen)
    and non-loopback Ollama/vLLM endpoints are permitted. Default: **off**.

``ASSISTANT_LOCAL_ENDPOINT_ALLOWLIST``
    Comma-separated hostnames treated as private/local in addition to loopback
    (e.g. ``llm.internal,10.0.0.5``). Empty by default.
"""

from __future__ import annotations

import os
import threading
from typing import Iterable, List, Optional, Sequence
from urllib.parse import urlparse

# Hosted third-party provider *names* (not class names).
HOSTED_PROVIDER_NAMES = frozenset(
    {
        "claude",
        "anthropic",
        "openai",
        "gpt",
        "gpt4",
        "qwen",
        "tongyi",
    }
)

# Names that are local only when their endpoint is loopback / allowlisted.
LOCAL_ENDPOINT_PROVIDER_NAMES = frozenset({"ollama", "local", "vllm"})

ALWAYS_LOCAL_PROVIDER_NAMES = frozenset({"offline"})

ENV_HOSTED_OPT_IN = "ASSISTANT_HOSTED_PROVIDER_OPT_IN"
ENV_LOCAL_ENDPOINT_ALLOWLIST = "ASSISTANT_LOCAL_ENDPOINT_ALLOWLIST"

# Attempt log for §6 discriminator #2 (zero hosted attempts on fallback).
_attempt_lock = threading.Lock()
_provider_attempts: List[str] = []


def clear_provider_attempts() -> None:
    """Test helper: reset the process-local attempt log."""
    with _attempt_lock:
        _provider_attempts.clear()


def get_provider_attempts() -> List[str]:
    """Return a copy of recorded provider generate/select attempts."""
    with _attempt_lock:
        return list(_provider_attempts)


def record_provider_attempt(provider_name: str) -> None:
    """Record that a concrete provider was selected for a network/generate attempt."""
    name = (provider_name or "unknown").lower()
    with _attempt_lock:
        _provider_attempts.append(name)


def hosted_provider_opt_in() -> bool:
    """True when the deployment explicitly opted into hosted LLM egress."""
    raw = os.getenv(ENV_HOSTED_OPT_IN, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def is_hosted_provider_name(provider_name: str) -> bool:
    return (provider_name or "").lower() in HOSTED_PROVIDER_NAMES


def _loopback_hosts() -> frozenset:
    return frozenset(
        {
            "localhost",
            "127.0.0.1",
            "::1",
            "0.0.0.0",
        }
    )


def local_endpoint_allowlist() -> frozenset:
    raw = os.getenv(ENV_LOCAL_ENDPOINT_ALLOWLIST, "") or ""
    parts = {p.strip().lower() for p in raw.split(",") if p.strip()}
    return frozenset(parts)


def endpoint_is_verified_local(url: str) -> bool:
    """Return True iff *url* targets loopback or an explicit allowlisted host.

    Empty / unparseable URLs are **not** local (fail-closed).
    """
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host in _loopback_hosts():
        return True
    if host in local_endpoint_allowlist():
        return True
    # RFC1918 private ranges — require explicit allowlist? Design says
    # "loopback or explicitly per-deployment-allowlisted private host".
    # Private IPs without allowlist are treated as external (fail-closed).
    return False


def default_endpoint_for_provider(provider_name: str) -> str:
    name = (provider_name or "").lower()
    if name in {"ollama", "local"}:
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if name == "vllm":
        return os.getenv("VLLM_ENDPOINT", "http://localhost:8100")
    return ""


def provider_name_requires_opt_in(provider_name: str) -> bool:
    """Whether selecting this provider name requires hosted opt-in right now.

    Hosted names always require opt-in. Ollama/vLLM require opt-in when their
    configured endpoint is not verified-local.
    """
    name = (provider_name or "").lower()
    if name in ALWAYS_LOCAL_PROVIDER_NAMES:
        return False
    if name in HOSTED_PROVIDER_NAMES:
        return True
    if name in LOCAL_ENDPOINT_PROVIDER_NAMES:
        endpoint = default_endpoint_for_provider(name)
        return not endpoint_is_verified_local(endpoint)
    # Unknown names: treat as hosted (fail-closed).
    return True


def resolve_provider_name(requested: str) -> str:
    """Map a requested provider name to a seal-safe name.

    Without opt-in, hosted and non-local endpoints resolve to ``offline``.
    """
    name = (requested or "offline").lower().strip() or "offline"
    if name in ALWAYS_LOCAL_PROVIDER_NAMES:
        return "offline"
    if provider_name_requires_opt_in(name) and not hosted_provider_opt_in():
        return "offline"
    return name


def sealed_fallback_chain(
    *,
    primary_name: Optional[str] = None,
) -> List[str]:
    """Return the ordered fallback chain under the SEAL contract.

    Hosted providers are **never** present in the fallback chain unless the
    deployment has opt-in **and** that exact hosted provider was the primary
    (in which case it is still skipped by the caller as already-failed).
    Other hosted providers are never attempted as failover.
    """
    # Local-only candidates; endpoint seal applied at resolve/get_provider time.
    chain = ["vllm", "ollama", "offline"]
    if not hosted_provider_opt_in():
        return chain
    # Opt-in deployments still must not cascade across hosted vendors.
    # Primary (if hosted) is excluded by the caller; we do not add claude/openai/qwen.
    return chain


def sealed_auto_select_order() -> List[str]:
    """Priority order for auto-select under the SEAL (no hosted without opt-in)."""
    if hosted_provider_opt_in():
        # Opt-in still prefers local first, then hosted — product can override via config.
        return ["vllm", "ollama", "claude", "openai", "qwen", "offline"]
    return ["vllm", "ollama", "offline"]


def filter_provider_names(names: Iterable[str]) -> List[str]:
    """Drop names that the seal would rewrite to offline without opt-in."""
    out: List[str] = []
    for n in names:
        resolved = resolve_provider_name(n)
        if resolved not in out:
            out.append(resolved)
    return out


__all__ = [
    "ALWAYS_LOCAL_PROVIDER_NAMES",
    "ENV_HOSTED_OPT_IN",
    "ENV_LOCAL_ENDPOINT_ALLOWLIST",
    "HOSTED_PROVIDER_NAMES",
    "LOCAL_ENDPOINT_PROVIDER_NAMES",
    "clear_provider_attempts",
    "default_endpoint_for_provider",
    "endpoint_is_verified_local",
    "filter_provider_names",
    "get_provider_attempts",
    "hosted_provider_opt_in",
    "is_hosted_provider_name",
    "local_endpoint_allowlist",
    "provider_name_requires_opt_in",
    "record_provider_attempt",
    "resolve_provider_name",
    "sealed_auto_select_order",
    "sealed_fallback_chain",
]
