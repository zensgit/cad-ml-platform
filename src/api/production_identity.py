"""Production identity fail-closed helpers (#517 design-lock runtime).

Canonical production rule (design-lock §1.B):
  Production UNLESS ENVIRONMENT/APP_ENV/ENV is exactly ``development`` or ``test``.
  Unset / unknown / conflicting → production (fail-closed).
  ``REQUIRE_STRONG_AUTH`` also forces production posture.

Dev/test permissive defaults require the explicit development opt-in (harness sets
``ENVIRONMENT=development`` in conftest and CI).
"""

from __future__ import annotations

import logging
import os
from typing import FrozenSet, Optional, Set

logger = logging.getLogger(__name__)

INSECURE_DEFAULT = "test"
_DEV_OPT_IN = frozenset({"development", "test"})
_TRUE = frozenset({"1", "true", "yes", "on"})


def deployment_env_raw() -> str:
    return (
        (os.getenv("ENVIRONMENT") or os.getenv("APP_ENV") or os.getenv("ENV") or "")
        .strip()
        .lower()
    )


def is_development_opt_in() -> bool:
    """True only when env is exactly development/test (explicit harness opt-in)."""
    return deployment_env_raw() in _DEV_OPT_IN


def is_production_posture() -> bool:
    """True when insecure defaults and disabled integration auth are forbidden."""
    if os.getenv("REQUIRE_STRONG_AUTH", "").strip().lower() in _TRUE:
        return True
    return not is_development_opt_in()


def configured_api_keys() -> Set[str]:
    """Expected API keys from env (API_KEYS comma-list, or API_KEY / X_API_KEY)."""
    raw = (
        os.getenv("API_KEYS") or os.getenv("API_KEY") or os.getenv("X_API_KEY") or ""
    ).strip()
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def configured_admin_token() -> str:
    return os.getenv("ADMIN_TOKEN", "").strip()


def expected_api_keys_for_request() -> Set[str]:
    """Keys that may authenticate a request under the current posture.

    Production: only non-empty configured keys that are not the insecure default.
    Development opt-in: configured keys, or ``{test}`` if none configured (harness).
    """
    keys = configured_api_keys()
    if is_production_posture():
        return {k for k in keys if k and k != INSECURE_DEFAULT}
    if not keys:
        return {INSECURE_DEFAULT}
    return keys


def validate_boot_identity(
    *,
    integration_auth_mode: str = "disabled",
    integration_jwt_secret: str = "",
    integration_jwt_audience: str = "",
    integration_jwt_issuer: str = "",
) -> Optional[str]:
    """Return an error string if production boot must refuse; else None."""
    mode = (integration_auth_mode or "disabled").strip().lower()
    secret = (integration_jwt_secret or "").strip()
    audience = (integration_jwt_audience or "").strip()
    issuer = (integration_jwt_issuer or "").strip()
    decisions_enabled = (
        os.getenv("REVIEW_REUSE_DECISIONS_ENABLED", "").strip().lower() in _TRUE
    )
    if decisions_enabled and (
        mode != "required" or not secret or not audience or not issuer
    ):
        return (
            "ReviewReuse decisions require INTEGRATION_AUTH_MODE=required and complete "
            "JWT secret, audience, and issuer configuration"
        )
    if not is_production_posture():
        return None

    keys = configured_api_keys()
    if not keys:
        return "production posture: API_KEY/API_KEYS unset — refuse to boot"
    if INSECURE_DEFAULT in keys:
        return f"production posture: API_KEY/API_KEYS must not include {INSECURE_DEFAULT!r}"

    admin = configured_admin_token()
    if not admin or admin == INSECURE_DEFAULT:
        return (
            "production posture: ADMIN_TOKEN unset or insecure default — refuse to boot"
        )

    if mode == "disabled":
        return "production posture: INTEGRATION_AUTH_MODE=disabled is not allowed — refuse to boot"

    if mode == "required":
        if not secret or not audience or not issuer:
            return (
                "production posture: INTEGRATION_AUTH_MODE=required requires "
                "INTEGRATION_JWT_SECRET, INTEGRATION_JWT_AUDIENCE, INTEGRATION_JWT_ISSUER"
            )
    if mode == "optional":
        # Any INTEGRATION_* partial config without secret is refuse.
        if (audience or issuer) and not secret:
            return (
                "production posture: INTEGRATION_JWT_AUDIENCE/ISSUER set without "
                "INTEGRATION_JWT_SECRET — refuse to boot"
            )

    return None


def refuse_boot_if_invalid(settings: object) -> None:
    """Call at process startup; raises SystemExit(1) on production misconfiguration."""
    mode = str(getattr(settings, "INTEGRATION_AUTH_MODE", "disabled") or "disabled")
    err = validate_boot_identity(
        integration_auth_mode=mode,
        integration_jwt_secret=str(
            getattr(settings, "INTEGRATION_JWT_SECRET", "") or ""
        ),
        integration_jwt_audience=str(
            getattr(settings, "INTEGRATION_JWT_AUDIENCE", "") or ""
        ),
        integration_jwt_issuer=str(
            getattr(settings, "INTEGRATION_JWT_ISSUER", "") or ""
        ),
    )
    if err:
        logger.error("identity boot refuse: %s", err)
        raise SystemExit(1)


__all__ = [
    "INSECURE_DEFAULT",
    "configured_admin_token",
    "configured_api_keys",
    "deployment_env_raw",
    "expected_api_keys_for_request",
    "is_development_opt_in",
    "is_production_posture",
    "refuse_boot_if_invalid",
    "validate_boot_identity",
]
