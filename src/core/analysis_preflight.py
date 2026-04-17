"""Shared request/file preflight helpers for analyze flows."""

from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from time import time as _time_now
from typing import Any, Awaitable, Callable, Dict, Optional

from src.utils.cache import get_cached_result

CacheGetter = Callable[[str], Awaitable[Optional[dict]]]


def _update_cache_window(*, hit: bool) -> None:
    try:
        if hit:
            from src.utils.analysis_metrics import feature_cache_hits_last_hour

            events = globals().setdefault("_CACHE_HIT_EVENTS", deque())
            metric = feature_cache_hits_last_hour
        else:
            from src.utils.analysis_metrics import feature_cache_miss_last_hour

            events = globals().setdefault("_CACHE_MISS_EVENTS", deque())
            metric = feature_cache_miss_last_hour

        now = _time_now()
        events.append(now)
        while events and now - events[0] > 3600:
            events.popleft()
        metric.set(len(events))
    except Exception:
        pass


def build_analysis_cache_key(
    *,
    file_name: Optional[str],
    content: bytes,
    options_raw: str,
) -> str:
    content_hash = hashlib.sha256(content).hexdigest()[:16]
    safe_file_name = file_name or ""
    analysis_cache_key = f"analysis:{safe_file_name}:{content_hash}:{options_raw}"

    include_part_shadow = os.getenv(
        "PART_CLASSIFIER_PROVIDER_INCLUDE_IN_CACHE_KEY", "true"
    ).strip().lower() not in {"0", "false", "no", "off"}
    if not include_part_shadow:
        return analysis_cache_key

    suffix = safe_file_name.rsplit(".", 1)[-1].lower()
    shadow_formats_raw = os.getenv("PART_CLASSIFIER_PROVIDER_SHADOW_FORMATS", "dxf,dwg")
    shadow_formats = {
        token.strip().lower() for token in shadow_formats_raw.split(",") if token.strip()
    }
    if suffix not in shadow_formats:
        return analysis_cache_key

    shadow_enabled = (
        os.getenv("PART_CLASSIFIER_PROVIDER_ENABLED", "false").strip().lower() == "true"
    )
    shadow_provider = os.getenv("PART_CLASSIFIER_PROVIDER_NAME", "v16").strip() or "v16"
    return (
        f"{analysis_cache_key}:part_shadow={int(shadow_enabled)}:{shadow_provider}"
    )


async def run_analysis_request_preflight(
    *,
    file_name: Optional[str],
    options_raw: str,
    content: bytes,
    analysis_id: str,
    timestamp: Any,
    options_model_cls: Any,
    cache_getter: Optional[CacheGetter] = None,
) -> Dict[str, Any]:
    analysis_options = options_model_cls(**json.loads(options_raw))
    analysis_cache_key = build_analysis_cache_key(
        file_name=file_name,
        content=content,
        options_raw=options_raw,
    )
    cached = await (cache_getter or get_cached_result)(analysis_cache_key)

    if cached is not None:
        from src.utils.analysis_metrics import analysis_cache_hits_total

        analysis_cache_hits_total.inc()
        _update_cache_window(hit=True)
        feature_version = os.getenv("FEATURE_VERSION", "v1")
        return {
            "analysis_options": analysis_options,
            "analysis_cache_key": analysis_cache_key,
            "cached": cached,
            "cached_response": {
                "id": analysis_id,
                "timestamp": timestamp,
                "file_name": file_name,
                "file_format": (file_name or "").split(".")[-1].upper(),
                "results": cached,
                "processing_time": 0.1,
                "cache_hit": True,
                "cad_document": None,
                "feature_version": feature_version,
            },
        }

    from src.utils.analysis_metrics import analysis_cache_miss_total

    analysis_cache_miss_total.inc()
    _update_cache_window(hit=False)
    return {
        "analysis_options": analysis_options,
        "analysis_cache_key": analysis_cache_key,
        "cached": None,
        "cached_response": None,
    }


__all__ = [
    "build_analysis_cache_key",
    "run_analysis_request_preflight",
]
