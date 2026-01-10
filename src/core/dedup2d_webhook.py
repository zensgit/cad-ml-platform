"""Dedup2D webhook callback utilities (Phase 3).

This module provides:
- Callback URL validation (basic SSRF mitigation, configurable by env)
- Optional HMAC signing for webhook payloads
- Async sender with retries/backoff
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

import httpx

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v == "":
        return default
    if v in _TRUE_VALUES:
        return True
    if v in _FALSE_VALUES:
        return False
    logger.warning("invalid_bool_env", extra={"name": name, "value": raw})
    return default


def _split_csv(value: str) -> list[str]:
    out: list[str] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(item)
    return out


@dataclass(frozen=True)
class Dedup2DWebhookConfig:
    allow_http: bool
    block_private_networks: bool
    resolve_dns: bool
    allowlist: list[str]
    hmac_secret: str
    timeout_seconds: float
    max_attempts: int
    backoff_base_seconds: float
    backoff_max_seconds: float

    @classmethod
    def from_env(cls) -> "Dedup2DWebhookConfig":
        allow_http = _env_bool("DEDUP2D_CALLBACK_ALLOW_HTTP", default=False)
        block_private = _env_bool("DEDUP2D_CALLBACK_BLOCK_PRIVATE_NETWORKS", default=True)
        resolve_dns = _env_bool("DEDUP2D_CALLBACK_RESOLVE_DNS", default=False)
        allowlist_raw = os.getenv("DEDUP2D_CALLBACK_ALLOWLIST", "").strip()
        allowlist = [h.lower() for h in _split_csv(allowlist_raw)] if allowlist_raw else []
        secret = os.getenv("DEDUP2D_CALLBACK_HMAC_SECRET", "").strip()
        timeout_seconds = float(os.getenv("DEDUP2D_CALLBACK_TIMEOUT_SECONDS", "10") or "10")
        max_attempts = int(os.getenv("DEDUP2D_CALLBACK_MAX_ATTEMPTS", "3") or "3")
        backoff_base_seconds = float(os.getenv("DEDUP2D_CALLBACK_BACKOFF_BASE_SECONDS", "1") or "1")
        backoff_max_seconds = float(os.getenv("DEDUP2D_CALLBACK_BACKOFF_MAX_SECONDS", "10") or "10")
        return cls(
            allow_http=allow_http,
            block_private_networks=block_private,
            resolve_dns=resolve_dns,
            allowlist=allowlist,
            hmac_secret=secret,
            timeout_seconds=max(0.1, timeout_seconds),
            max_attempts=max(1, max_attempts),
            backoff_base_seconds=max(0.0, backoff_base_seconds),
            backoff_max_seconds=max(0.0, backoff_max_seconds),
        )


def _is_ip_allowed(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return False
    if ip.is_private:
        return False
    return True


def validate_dedup2d_callback_url(url: str, *, cfg: Optional[Dedup2DWebhookConfig] = None) -> str:
    """Validate callback URL for basic SSRF protection.

    - Enforces scheme (https by default; http optionally for dev)
    - Blocks userinfo in URL (username/password)
    - Optional allowlist by hostname
    - Optional private network blocking (IP literals; and DNS resolution if enabled)
    """
    effective_cfg = cfg or Dedup2DWebhookConfig.from_env()
    raw = str(url or "").strip()
    if not raw:
        raise ValueError("callback_url is empty")

    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    allowed_schemes = {"https"}
    if effective_cfg.allow_http:
        allowed_schemes.add("http")
    if scheme not in allowed_schemes:
        raise ValueError(f"callback_url scheme not allowed: {scheme}")
    if parsed.username or parsed.password:
        raise ValueError("callback_url must not include userinfo")
    if not parsed.hostname:
        raise ValueError("callback_url missing hostname")

    host = parsed.hostname.lower()
    if effective_cfg.allowlist and host not in effective_cfg.allowlist:
        raise ValueError("callback_url hostname not in allowlist")

    if effective_cfg.block_private_networks:
        # Fast path: IP literals
        ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            ip = None
        if ip is not None:
            if not _is_ip_allowed(ip):
                raise ValueError("callback_url hostname resolves to private network") from None
        elif effective_cfg.resolve_dns:
            port = parsed.port
            if port is None:
                port = 443 if scheme == "https" else 80
            try:
                infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            except OSError as e:
                raise ValueError(f"callback_url DNS resolution failed: {e}") from e
            for info in infos:
                addr = info[4][0]
                try:
                    ip_resolved = ipaddress.ip_address(addr)
                except ValueError:
                    continue
                if not _is_ip_allowed(ip_resolved):
                    raise ValueError("callback_url hostname resolves to private network") from None

    # Normalize: drop fragment, ensure path present
    normalized_path = parsed.path or "/"
    normalized = urlunparse(
        (
            scheme,
            parsed.netloc,
            normalized_path,
            parsed.params,
            parsed.query,
            "",
        )
    )
    return normalized


def sign_dedup2d_webhook(
    *,
    secret: str,
    job_id: str,
    body: bytes,
    timestamp: Optional[int] = None,
) -> Dict[str, str]:
    ts = int(timestamp or int(time.time()))
    msg = f"{ts}.{job_id}.".encode("utf-8") + body
    sig = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
    return {
        "X-Dedup-Signature": f"t={ts},v1={sig}",
        "X-Dedup-Job-Id": str(job_id),
        "X-Dedup-Signature-Version": "v1",
    }


def _is_retryable_status(code: int) -> bool:
    if code in {408, 425, 429}:
        return True
    if 500 <= code <= 599:
        return True
    return False


async def send_dedup2d_webhook(
    *,
    callback_url: str,
    payload: Dict[str, Any],
    job_id: str,
    tenant_id: Optional[str],
    cfg: Optional[Dedup2DWebhookConfig] = None,
) -> tuple[bool, int, Optional[int], Optional[str]]:
    """Send a webhook payload to callback_url.

    Returns: (success, attempts, http_status, error_message)
    """
    effective_cfg = cfg or Dedup2DWebhookConfig.from_env()
    normalized = validate_dedup2d_callback_url(callback_url, cfg=effective_cfg)

    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": "cad-ml-platform/dedup2d-webhook",
        "X-Dedup-Job-Id": str(job_id),
    }
    if tenant_id:
        headers["X-Dedup-Tenant-Id"] = str(tenant_id)
    if effective_cfg.hmac_secret:
        headers.update(
            sign_dedup2d_webhook(
                secret=effective_cfg.hmac_secret,
                job_id=job_id,
                body=body,
            )
        )

    timeout = httpx.Timeout(effective_cfg.timeout_seconds)
    attempts = 0
    backoff = float(effective_cfg.backoff_base_seconds)
    last_error: Optional[str] = None
    last_status: Optional[int] = None

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        for attempts in range(1, int(effective_cfg.max_attempts) + 1):
            try:
                resp = await client.post(normalized, content=body, headers=headers)
                last_status = int(resp.status_code)
                if 200 <= last_status <= 299:
                    return True, attempts, last_status, None

                last_error = f"http_status:{last_status}"
                if not _is_retryable_status(last_status):
                    return False, attempts, last_status, last_error
            except httpx.RequestError as e:
                last_error = str(e)
                last_status = None

            if attempts < int(effective_cfg.max_attempts) and backoff > 0:
                await asyncio.sleep(backoff)
                backoff = min(max(backoff, 0.0) * 2.0, float(effective_cfg.backoff_max_seconds))

    return False, attempts, last_status, last_error


__all__ = [
    "Dedup2DWebhookConfig",
    "send_dedup2d_webhook",
    "sign_dedup2d_webhook",
    "validate_dedup2d_callback_url",
]
