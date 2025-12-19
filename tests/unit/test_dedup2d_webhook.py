from __future__ import annotations

import pytest

from src.core.dedup2d_webhook import (
    Dedup2DWebhookConfig,
    sign_dedup2d_webhook,
    validate_dedup2d_callback_url,
)


def test_validate_callback_url_requires_https_by_default():
    cfg = Dedup2DWebhookConfig(
        allow_http=False,
        block_private_networks=True,
        resolve_dns=False,
        allowlist=[],
        hmac_secret="",
        timeout_seconds=1.0,
        max_attempts=1,
        backoff_base_seconds=0.0,
        backoff_max_seconds=0.0,
    )
    assert validate_dedup2d_callback_url("https://example.com/hook", cfg=cfg).startswith(
        "https://"
    )
    with pytest.raises(ValueError):
        validate_dedup2d_callback_url("http://example.com/hook", cfg=cfg)


def test_validate_callback_url_blocks_userinfo_and_private_ip_literals():
    cfg = Dedup2DWebhookConfig(
        allow_http=True,
        block_private_networks=True,
        resolve_dns=False,
        allowlist=[],
        hmac_secret="",
        timeout_seconds=1.0,
        max_attempts=1,
        backoff_base_seconds=0.0,
        backoff_max_seconds=0.0,
    )
    with pytest.raises(ValueError):
        validate_dedup2d_callback_url("https://user:pass@example.com/hook", cfg=cfg)
    with pytest.raises(ValueError):
        validate_dedup2d_callback_url("http://127.0.0.1/callback", cfg=cfg)


def test_validate_callback_url_allowlist_enforced():
    cfg = Dedup2DWebhookConfig(
        allow_http=False,
        block_private_networks=False,
        resolve_dns=False,
        allowlist=["hooks.example.com"],
        hmac_secret="",
        timeout_seconds=1.0,
        max_attempts=1,
        backoff_base_seconds=0.0,
        backoff_max_seconds=0.0,
    )
    assert validate_dedup2d_callback_url("https://hooks.example.com/x", cfg=cfg).startswith(
        "https://hooks.example.com"
    )
    with pytest.raises(ValueError):
        validate_dedup2d_callback_url("https://example.com/x", cfg=cfg)


def test_sign_webhook_has_expected_headers():
    body = b'{"ok":true}'
    headers = sign_dedup2d_webhook(secret="secret", job_id="job1", body=body, timestamp=123)
    assert headers["X-Dedup-Job-Id"] == "job1"
    assert headers["X-Dedup-Signature"].startswith("t=123,v1=")

