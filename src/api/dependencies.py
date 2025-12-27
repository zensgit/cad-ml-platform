"""API dependency stubs for testing environment."""

import os

from fastapi import Header, HTTPException


async def get_api_key(x_api_key: str = Header(default="test", alias="X-API-Key")) -> str:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API Key")
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

    # Get expected token from environment; default to 'test' for dev/test
    expected_token = os.getenv("ADMIN_TOKEN", "test")

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
