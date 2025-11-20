"""Idempotency key support for API endpoints.

Ensures that duplicate requests with the same idempotency key
return the same cached response without re-processing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.utils.cache import get_cache, set_cache

logger = logging.getLogger(__name__)

# Idempotency cache TTL (24 hours to cover retries)
IDEMPOTENCY_TTL_SECONDS = 86400


def build_idempotency_key(idempotency_key: str, endpoint: str = "ocr") -> str:
    """Build Redis key for idempotency storage.

    Args:
        idempotency_key: Client-provided idempotency key
        endpoint: API endpoint name

    Returns:
        Redis cache key in format: idempotency:{endpoint}:{key}
    """
    return f"idempotency:{endpoint}:{idempotency_key}"


async def check_idempotency(
    idempotency_key: str, endpoint: str = "ocr"
) -> Optional[Dict[str, Any]]:
    """Check if response exists for given idempotency key.

    Args:
        idempotency_key: Client-provided idempotency key
        endpoint: API endpoint name

    Returns:
        Cached response if exists, None otherwise
    """
    if not idempotency_key:
        return None

    cache_key = build_idempotency_key(idempotency_key, endpoint)
    cached = await get_cache(cache_key)

    if cached:
        logger.info(
            "idempotency.cache_hit",
            extra={
                "idempotency_key": idempotency_key,
                "endpoint": endpoint,
            },
        )

    return cached


async def store_idempotency(
    idempotency_key: str,
    response: Dict[str, Any],
    endpoint: str = "ocr",
    ttl_seconds: int = IDEMPOTENCY_TTL_SECONDS,
) -> None:
    """Store response for idempotency key.

    Args:
        idempotency_key: Client-provided idempotency key
        response: Response data to cache
        endpoint: API endpoint name
        ttl_seconds: Time-to-live for cached response
    """
    if not idempotency_key:
        return

    cache_key = build_idempotency_key(idempotency_key, endpoint)
    await set_cache(cache_key, response, ttl_seconds)

    logger.info(
        "idempotency.stored",
        extra={
            "idempotency_key": idempotency_key,
            "endpoint": endpoint,
            "ttl_seconds": ttl_seconds,
        },
    )
