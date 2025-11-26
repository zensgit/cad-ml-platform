"""API dependency stubs for testing environment."""

import os
from fastapi import Header, HTTPException


async def get_api_key(x_api_key: str = Header(default="test", alias="X-API-Key")) -> str:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API Key")
    return x_api_key


async def get_admin_token(x_admin_token: str = Header(alias="X-Admin-Token")) -> str:
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
            {"hint": "Provide X-Admin-Token header"}
        )
        raise HTTPException(status_code=401, detail=error)

    # Get expected token from environment
    expected_token = os.getenv("ADMIN_TOKEN")
    if not expected_token:
        from src.core.errors_extended import ErrorCode, create_extended_error
        error = create_extended_error(
            ErrorCode.AUTHORIZATION_FAILED,
            "Admin token not configured",
            {"hint": "Set ADMIN_TOKEN environment variable"}
        )
        raise HTTPException(status_code=500, detail=error)

    # Validate token
    if x_admin_token != expected_token:
        from src.core.errors_extended import ErrorCode, create_extended_error
        error = create_extended_error(
            ErrorCode.AUTHORIZATION_FAILED,
            "Invalid admin token",
            {"hint": "Check X-Admin-Token header"}
        )
        raise HTTPException(status_code=403, detail=error)

    return x_admin_token
