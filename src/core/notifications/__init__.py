"""Deployment Notification System.

Provides multi-channel notifications for deployment events:
- Slack
- DingTalk
- WeChat Work
- Generic Webhooks
"""

from src.core.notifications.client import (
    NotificationClient,
    NotificationChannel,
    NotificationMessage,
    NotificationPriority,
    get_notification_client,
)
from src.core.notifications.channels import (
    SlackChannel,
    DingTalkChannel,
    WeChatChannel,
    WebhookChannel,
)

__all__ = [
    "NotificationClient",
    "NotificationChannel",
    "NotificationMessage",
    "NotificationPriority",
    "get_notification_client",
    "SlackChannel",
    "DingTalkChannel",
    "WeChatChannel",
    "WebhookChannel",
]
