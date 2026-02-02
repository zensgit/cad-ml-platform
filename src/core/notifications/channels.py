"""Notification Channel Implementations."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
import urllib.parse
from base64 import b64encode
from typing import Any, Dict, Optional

import httpx

from .client import NotificationChannel, NotificationMessage, NotificationPriority

logger = logging.getLogger(__name__)


class SlackChannel(NotificationChannel):
    """Slack notification channel."""

    name = "slack"

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        channel: Optional[str] = None,
        username: str = "CAD ML Platform",
        icon_emoji: str = ":robot_face:",
    ):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL", "")
        self.channel = channel or os.getenv("SLACK_CHANNEL", "")
        self.username = username
        self.icon_emoji = icon_emoji

    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    def _get_color(self, message: NotificationMessage) -> str:
        """Get Slack attachment color based on event type."""
        if "failed" in message.event_type.value or "degraded" in message.event_type.value:
            return "#E96D76"  # Red
        elif "succeeded" in message.event_type.value or "completed" in message.event_type.value:
            return "#18be52"  # Green
        elif "rollback" in message.event_type.value:
            return "#f4c030"  # Yellow
        return "#36a64f"  # Default green

    def _build_payload(self, message: NotificationMessage) -> Dict[str, Any]:
        """Build Slack message payload."""
        fields = []

        if message.environment:
            fields.append({"title": "Environment", "value": message.environment, "short": True})
        if message.version:
            fields.append({"title": "Version", "value": message.version, "short": True})
        if message.service:
            fields.append({"title": "Service", "value": message.service, "short": True})
        if message.error_message:
            fields.append({"title": "Error", "value": message.error_message[:200], "short": False})

        payload = {
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "attachments": [
                {
                    "color": self._get_color(message),
                    "title": message.title,
                    "text": message.body,
                    "fields": fields,
                    "footer": "CAD ML Platform",
                    "ts": int(message.timestamp.timestamp()),
                }
            ],
        }

        if self.channel:
            payload["channel"] = self.channel

        return payload

    async def send(self, message: NotificationMessage) -> bool:
        """Send Slack notification."""
        try:
            payload = self._build_payload(message)
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10.0,
                )
                if response.status_code == 200:
                    logger.info(f"Slack notification sent: {message.title}")
                    return True
                else:
                    logger.error(f"Slack API error: {response.status_code} - {response.text}")
                    return False
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False


class DingTalkChannel(NotificationChannel):
    """DingTalk (钉钉) notification channel."""

    name = "dingtalk"

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        secret: Optional[str] = None,
    ):
        self.webhook_url = webhook_url or os.getenv("DINGTALK_WEBHOOK_URL", "")
        self.secret = secret or os.getenv("DINGTALK_SECRET", "")

    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    def _get_signed_url(self) -> str:
        """Get signed webhook URL with timestamp and signature."""
        if not self.secret:
            return self.webhook_url

        timestamp = str(int(time.time() * 1000))
        string_to_sign = f"{timestamp}\n{self.secret}"
        hmac_code = hmac.new(
            self.secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        sign = urllib.parse.quote_plus(b64encode(hmac_code).decode("utf-8"))

        return f"{self.webhook_url}&timestamp={timestamp}&sign={sign}"

    def _get_emoji(self, message: NotificationMessage) -> str:
        """Get emoji based on event type."""
        if "failed" in message.event_type.value:
            return "❌"
        elif "succeeded" in message.event_type.value:
            return "✅"
        elif "rollback" in message.event_type.value:
            return "⏪"
        elif "degraded" in message.event_type.value:
            return "⚠️"
        elif "started" in message.event_type.value:
            return "🚀"
        return "📢"

    def _build_payload(self, message: NotificationMessage) -> Dict[str, Any]:
        """Build DingTalk message payload."""
        emoji = self._get_emoji(message)

        # Build markdown content
        content_lines = [
            f"### {emoji} {message.title}",
            "",
            message.body,
            "",
        ]

        if message.environment:
            content_lines.append(f"**环境:** {message.environment}")
        if message.version:
            content_lines.append(f"**版本:** {message.version}")
        if message.service:
            content_lines.append(f"**服务:** {message.service}")
        if message.error_message:
            content_lines.append(f"**错误:** {message.error_message[:200]}")

        content_lines.extend([
            "",
            f"*时间: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC*",
        ])

        payload = {
            "msgtype": "markdown",
            "markdown": {
                "title": message.title,
                "text": "\n".join(content_lines),
            },
        }

        # Add @all for urgent messages
        if message.priority == NotificationPriority.URGENT:
            payload["at"] = {"isAtAll": True}

        return payload

    async def send(self, message: NotificationMessage) -> bool:
        """Send DingTalk notification."""
        try:
            url = self._get_signed_url()
            payload = self._build_payload(message)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=payload,
                    timeout=10.0,
                )
                result = response.json()
                if result.get("errcode") == 0:
                    logger.info(f"DingTalk notification sent: {message.title}")
                    return True
                else:
                    logger.error(f"DingTalk API error: {result}")
                    return False
        except Exception as e:
            logger.error(f"Failed to send DingTalk notification: {e}")
            return False


class WeChatChannel(NotificationChannel):
    """WeChat Work (企业微信) notification channel."""

    name = "wechat"

    def __init__(
        self,
        webhook_url: Optional[str] = None,
    ):
        self.webhook_url = webhook_url or os.getenv("WECHAT_WEBHOOK_URL", "")

    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    def _build_payload(self, message: NotificationMessage) -> Dict[str, Any]:
        """Build WeChat Work message payload."""
        # Build markdown content
        content_lines = [
            f"## {message.title}",
            "",
            message.body,
            "",
        ]

        if message.environment:
            content_lines.append(f"> 环境: <font color=\"info\">{message.environment}</font>")
        if message.version:
            content_lines.append(f"> 版本: <font color=\"info\">{message.version}</font>")
        if message.service:
            content_lines.append(f"> 服务: <font color=\"info\">{message.service}</font>")
        if message.error_message:
            content_lines.append(f"> 错误: <font color=\"warning\">{message.error_message[:200]}</font>")

        return {
            "msgtype": "markdown",
            "markdown": {
                "content": "\n".join(content_lines),
            },
        }

    async def send(self, message: NotificationMessage) -> bool:
        """Send WeChat Work notification."""
        try:
            payload = self._build_payload(message)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10.0,
                )
                result = response.json()
                if result.get("errcode") == 0:
                    logger.info(f"WeChat notification sent: {message.title}")
                    return True
                else:
                    logger.error(f"WeChat API error: {result}")
                    return False
        except Exception as e:
            logger.error(f"Failed to send WeChat notification: {e}")
            return False


class WebhookChannel(NotificationChannel):
    """Generic webhook notification channel."""

    name = "webhook"

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        channel_name: str = "webhook",
    ):
        self.webhook_url = webhook_url or os.getenv("NOTIFICATION_WEBHOOK_URL", "")
        self.headers = headers or {}
        self.name = channel_name

        # Add default headers
        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/json"

    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    async def send(self, message: NotificationMessage) -> bool:
        """Send webhook notification."""
        try:
            payload = message.to_dict()

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=10.0,
                )
                if response.status_code in (200, 201, 202, 204):
                    logger.info(f"Webhook notification sent to {self.name}: {message.title}")
                    return True
                else:
                    logger.error(f"Webhook error ({self.name}): {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Failed to send webhook notification ({self.name}): {e}")
            return False


def setup_default_channels(client: Any) -> None:
    """Setup default notification channels from environment."""
    from .client import NotificationClient, NotificationEventType

    if not isinstance(client, NotificationClient):
        return

    # Register Slack
    slack = SlackChannel()
    if slack.is_configured():
        client.register_channel(slack, default=True)

    # Register DingTalk
    dingtalk = DingTalkChannel()
    if dingtalk.is_configured():
        client.register_channel(dingtalk, default=True)

    # Register WeChat
    wechat = WeChatChannel()
    if wechat.is_configured():
        client.register_channel(wechat, default=True)

    # Subscribe high-priority events to all channels
    for event_type in [
        NotificationEventType.DEPLOY_FAILED,
        NotificationEventType.ROLLBACK_COMPLETED,
        NotificationEventType.HEALTH_DEGRADED,
    ]:
        for channel_name in client._channels:
            client.subscribe(event_type, channel_name)
