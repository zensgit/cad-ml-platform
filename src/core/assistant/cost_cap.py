"""Fail-closed external-AI cost cap and isolated-sample mode (§8.3 pilot gates).

PRODUCT_STRATEGY.md §8.3 requires, before external AI is enabled:
  - provider spend exposure / budget, and a **fail-closed cost cap**;
  - isolated sample handling with **no external-provider egress**.

This module is the enforcement seam used before hosted LLM providers are
constructed. It does not mint retrain unlocks or touch Track E metrics (H).
"""

from __future__ import annotations

import os
from typing import Optional

# Opt-in for hosted LLM egress (existing seal).
_ENV_HOSTED_OPT_IN = "ASSISTANT_HOSTED_PROVIDER_OPT_IN"
# Required positive USD cap whenever hosted opt-in is true.
ENV_EXTERNAL_AI_COST_CAP_USD = "EXTERNAL_AI_COST_CAP_USD"
# Optional observed spend (defaults to 0). Callers / metering may update this.
ENV_EXTERNAL_AI_SPEND_USD = "EXTERNAL_AI_SPEND_USD"
# Isolated offline sample path: no external AI regardless of opt-in flags.
ENV_ISOLATED_SAMPLE_MODE = "ISOLATED_SAMPLE_MODE"

_TRUE = frozenset({"1", "true", "yes", "on"})


class CostCapRejected(RuntimeError):
    """Raised when external AI must not proceed under §8.3 fail-closed rules."""


def _truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE


def isolated_sample_mode() -> bool:
    """True when legally obtained samples must stay offline (no provider egress)."""
    return _truthy(ENV_ISOLATED_SAMPLE_MODE)


def hosted_opt_in() -> bool:
    return _truthy(_ENV_HOSTED_OPT_IN)


def parse_usd(raw: Optional[str], *, field: str) -> float:
    text = (raw or "").strip()
    if not text:
        raise CostCapRejected(f"{field} is required and must be a positive USD amount")
    try:
        value = float(text)
    except ValueError as exc:
        raise CostCapRejected(f"{field} must be a finite USD number, got {text!r}") from exc
    if value != value or value in (float("inf"), float("-inf")):  # NaN/inf
        raise CostCapRejected(f"{field} must be a finite USD number, got {text!r}")
    return value


def current_spend_usd() -> float:
    raw = os.getenv(ENV_EXTERNAL_AI_SPEND_USD, "0")
    try:
        spend = float((raw or "0").strip() or "0")
    except ValueError as exc:
        raise CostCapRejected(
            f"{ENV_EXTERNAL_AI_SPEND_USD} must be a finite USD number, got {raw!r}"
        ) from exc
    if spend != spend or spend < 0 or spend in (float("inf"), float("-inf")):
        raise CostCapRejected(
            f"{ENV_EXTERNAL_AI_SPEND_USD} must be a non-negative finite USD number"
        )
    return spend


def cost_cap_usd() -> float:
    """Return configured cap; raises if missing/invalid when required."""
    cap = parse_usd(os.getenv(ENV_EXTERNAL_AI_COST_CAP_USD), field=ENV_EXTERNAL_AI_COST_CAP_USD)
    if cap <= 0:
        raise CostCapRejected(
            f"{ENV_EXTERNAL_AI_COST_CAP_USD} must be > 0 when external AI is enabled"
        )
    return cap


def _is_production_posture() -> bool:
    """Align with production_identity: unset env = production fail-closed."""
    try:
        from src.api.production_identity import is_production_posture

        return bool(is_production_posture())
    except Exception:
        # If identity helpers are unavailable, treat as production (fail-closed).
        env = (
            os.getenv("ENVIRONMENT") or os.getenv("APP_ENV") or os.getenv("ENV") or ""
        ).strip().lower()
        return env not in {"development", "test"}


def assert_external_ai_allowed(*, provider_name: str = "") -> None:
    """Fail-closed gate before a hosted/external LLM provider may be used.

    Rules:
    1. ``ISOLATED_SAMPLE_MODE`` → always refuse external AI (any environment).
    2. Hosted opt-in false → callers seal to offline; this is only invoked for
       sealed hosted names.
    3. **Production posture** + hosted provider → require positive
       ``EXTERNAL_AI_COST_CAP_USD`` and ``EXTERNAL_AI_SPEND_USD < cap``.
    4. **Development/test harness** → if cap is set, enforce spend < cap; if cap
       is unset, allow construction so unit tests / local opt-in still work.
       Pilot enablement still requires production posture + explicit cap.
    """
    if isolated_sample_mode():
        raise CostCapRejected(
            "ISOLATED_SAMPLE_MODE is enabled: external AI / provider egress is forbidden "
            "(PRODUCT_STRATEGY §8.3 isolated sample handling)"
        )

    raw_cap = os.getenv(ENV_EXTERNAL_AI_COST_CAP_USD)
    production = _is_production_posture()
    if production or (raw_cap is not None and str(raw_cap).strip() != ""):
        # Production always requires a positive cap; dev only enforces when set.
        if production and (raw_cap is None or str(raw_cap).strip() == ""):
            raise CostCapRejected(
                f"{ENV_EXTERNAL_AI_COST_CAP_USD} is required in production posture "
                "before external AI is enabled (PRODUCT_STRATEGY §8.3)"
            )
        cap = cost_cap_usd()
        spend = current_spend_usd()
        if spend >= cap:
            raise CostCapRejected(
                f"external AI spend {spend} USD is at or above cost cap {cap} USD "
                f"(provider={provider_name or 'hosted'})"
            )


__all__ = [
    "CostCapRejected",
    "ENV_EXTERNAL_AI_COST_CAP_USD",
    "ENV_EXTERNAL_AI_SPEND_USD",
    "ENV_ISOLATED_SAMPLE_MODE",
    "assert_external_ai_allowed",
    "cost_cap_usd",
    "current_spend_usd",
    "hosted_opt_in",
    "isolated_sample_mode",
]
