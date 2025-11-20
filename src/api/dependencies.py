"""API dependency stubs for testing environment."""

from fastapi import Header, HTTPException


async def get_api_key(x_api_key: str = Header(default="test", alias="X-API-Key")) -> str:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API Key")
    return x_api_key
