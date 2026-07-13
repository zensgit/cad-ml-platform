"""API dependency stubs for testing environment."""

import os

from fastapi import Header, HTTPException


_INSECURE_DEFAULT = "test"


def _production_posture() -> bool:
    """True when the deployment must NOT accept the insecure ``test`` default (fail-closed).

    Triggered by an explicit ``REQUIRE_STRONG_AUTH`` flag, or by a real deployment environment
    (``ENVIRONMENT``/``APP_ENV``/``ENV`` in production/prod/staging). Dev and CI leave these unset,
    so the historical ``test`` default is preserved there and the suite is not bricked.
    """
    if os.getenv("REQUIRE_STRONG_AUTH", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    env = (
        os.getenv("ENVIRONMENT")
        or os.getenv("APP_ENV")
        or os.getenv("ENV")
        or ""
    ).strip().lower()
    return env in {"production", "prod", "staging"}


async def get_api_key(x_api_key: str = Header(default="test", alias="X-API-Key")) -> str:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API Key")
    # Fail-closed in a production posture: the literal insecure default must not authenticate.
    if _production_posture() and x_api_key == _INSECURE_DEFAULT:
        raise HTTPException(
            status_code=401,
            detail="Default 'test' API key is not accepted in a production posture (set a real X-API-Key).",
        )
    expected = os.getenv("API_KEY", "").strip()
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


async def get_admin_token(x_admin_token: str = Header(default="", alias="X-Admin-Token")) -> str:
    """
    验证管理员令牌，用于后端管理操作（模型重载等）

    Args:
        x_admin_token: 管理员令牌（从环境变量 ADMIN_TOKEN 中获取）

    Returns:
        验证通过的令牌

    Raises:
        HTTPException: 401 如果令牌缺失，403 如果令牌无效
    """
    if not x_admin_token:
        from src.core.errors_extended import ErrorCode, create_extended_error

        error = create_extended_error(
            ErrorCode.AUTHORIZATION_FAILED,
            "Missing admin token",
            context={"hint": "Provide X-Admin-Token header"},
        )
        raise HTTPException(status_code=401, detail=error.to_dict())

    # Fail-closed in a production posture: refuse to OPERATE with an unset or default admin token,
    # rather than silently trusting the literal "test". Dev/CI (no prod posture) keep the default.
    configured = os.getenv("ADMIN_TOKEN", "").strip()
    if _production_posture():
        if not configured or configured == _INSECURE_DEFAULT:
            from src.core.errors_extended import ErrorCode, create_extended_error

            error = create_extended_error(
                ErrorCode.AUTHORIZATION_FAILED,
                "ADMIN_TOKEN is unset or the insecure default in a production posture — refusing (fail-closed)",
                context={"hint": "Set ADMIN_TOKEN to a strong secret"},
            )
            raise HTTPException(status_code=500, detail=error.to_dict())
        expected_token = configured
    else:
        expected_token = configured or _INSECURE_DEFAULT  # dev/test convenience default preserved

    # Validate token
    if x_admin_token != expected_token:
        from src.core.errors_extended import ErrorCode, create_extended_error

        error = create_extended_error(
            ErrorCode.AUTHORIZATION_FAILED,
            "Invalid admin token",
            context={"hint": "Check X-Admin-Token header"},
        )
        raise HTTPException(status_code=403, detail=error.to_dict())

    return x_admin_token
