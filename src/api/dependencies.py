"""API authentication dependencies (production identity fail-closed)."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import Header, HTTPException

from src.api.production_identity import (
    INSECURE_DEFAULT,
    configured_admin_token,
    expected_api_keys_for_request,
    is_production_posture,
)


async def get_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> str:
    """Authenticate ``X-API-Key`` against the configured expected key set.

    Design-lock §1.A: no default value authenticates. Unset / unknown / insecure
    default → 401. Development opt-in may accept the harness key ``test`` when
    no ``API_KEY`` is configured (see :func:`expected_api_keys_for_request`).
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API Key")

    expected = expected_api_keys_for_request()
    if not expected or x_api_key not in expected:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # Defense in depth: production never accepts the insecure default even if
    # mis-listed (boot refuse should already have blocked that config).
    if is_production_posture() and x_api_key == INSECURE_DEFAULT:
        raise HTTPException(
            status_code=401,
            detail=(
                "Default 'test' API key is not accepted in a production posture "
                "(set API_KEY/API_KEYS to a real secret)."
            ),
        )
    return x_api_key


async def get_admin_token(
    x_admin_token: str = Header(default="", alias="X-Admin-Token"),
) -> str:
    """Validate admin token for privileged operations."""
    if not x_admin_token:
        from src.core.errors_extended import ErrorCode, create_extended_error

        error = create_extended_error(
            ErrorCode.AUTHORIZATION_FAILED,
            "Missing admin token",
            context={"hint": "Provide X-Admin-Token header"},
        )
        raise HTTPException(status_code=401, detail=error.to_dict())

    configured = configured_admin_token()
    if is_production_posture():
        if not configured or configured == INSECURE_DEFAULT:
            from src.core.errors_extended import ErrorCode, create_extended_error

            error = create_extended_error(
                ErrorCode.AUTHORIZATION_FAILED,
                "ADMIN_TOKEN is unset or the insecure default in a production "
                "posture — refusing (fail-closed)",
                context={"hint": "Set ADMIN_TOKEN to a strong secret"},
            )
            raise HTTPException(status_code=500, detail=error.to_dict())
        expected_token = configured
    else:
        # Development opt-in: harness default preserved when ADMIN_TOKEN unset.
        expected_token = configured or INSECURE_DEFAULT

    if x_admin_token != expected_token:
        from src.core.errors_extended import ErrorCode, create_extended_error

        error = create_extended_error(
            ErrorCode.AUTHORIZATION_FAILED,
            "Invalid admin token",
            context={"hint": "Check X-Admin-Token header"},
        )
        raise HTTPException(status_code=403, detail=error.to_dict())

    return x_admin_token


# Back-compat for bleeding-control tests that imported the private helper name.
def _production_posture() -> bool:
    return is_production_posture()
