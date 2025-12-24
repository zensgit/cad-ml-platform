"""Integration Hub Module for Vision System.

This module provides third-party integration capabilities including:
- Connector registry and management
- Webhook handling and event distribution
- OAuth/API key management
- Rate limiting for external services
- Data transformation and mapping
- Integration health monitoring

Phase 16: Advanced Integration & Extensibility
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from .base import VisionDescription, VisionProvider

# ========================
# Enums
# ========================


class ConnectorType(str, Enum):
    """Types of connectors."""

    REST_API = "rest_api"
    GRAPHQL = "graphql"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    DATABASE = "database"
    MESSAGE_QUEUE = "message_queue"
    FILE_STORAGE = "file_storage"
    STREAMING = "streaming"
    CUSTOM = "custom"


class AuthenticationType(str, Enum):
    """Authentication types."""

    NONE = "none"
    API_KEY = "api_key"
    BASIC = "basic"
    BEARER = "bearer"
    OAUTH2 = "oauth2"
    OAUTH2_CLIENT = "oauth2_client"
    JWT = "jwt"
    CUSTOM = "custom"


class ConnectorStatus(str, Enum):
    """Connector status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    CONNECTING = "connecting"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"


class WebhookEventType(str, Enum):
    """Webhook event types."""

    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"
    PROVIDER_CONNECTED = "provider.connected"
    PROVIDER_DISCONNECTED = "provider.disconnected"
    RATE_LIMIT_REACHED = "rate_limit.reached"
    ERROR_OCCURRED = "error.occurred"
    HEALTH_CHECK = "health.check"
    CUSTOM = "custom"


class DataFormat(str, Enum):
    """Data formats for transformation."""

    JSON = "json"
    XML = "xml"
    CSV = "csv"
    YAML = "yaml"
    PROTOBUF = "protobuf"
    MSGPACK = "msgpack"
    RAW = "raw"


# ========================
# Data Classes
# ========================


@dataclass
class AuthCredentials:
    """Authentication credentials."""

    auth_type: AuthenticationType
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    refresh_token: Optional[str] = None
    token_url: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    custom_headers: Dict[str, str] = field(default_factory=dict)
    expires_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if credentials are expired."""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_second: float = 10.0
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    burst_size: int = 10
    retry_after: int = 60  # seconds
    enabled: bool = True


@dataclass
class ConnectorConfig:
    """Connector configuration."""

    connector_id: str
    name: str
    connector_type: ConnectorType
    base_url: str
    auth: Optional[AuthCredentials] = None
    rate_limit: Optional[RateLimitConfig] = None
    timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class WebhookConfig:
    """Webhook configuration."""

    webhook_id: str
    url: str
    events: List[WebhookEventType]
    secret: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    retry_count: int = 3
    timeout: float = 10.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class WebhookPayload:
    """Webhook event payload."""

    event_type: WebhookEventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    webhook_id: Optional[str] = None
    signature: Optional[str] = None


@dataclass
class IntegrationHealth:
    """Integration health status."""

    connector_id: str
    status: ConnectorStatus
    last_check: datetime
    response_time_ms: float = 0.0
    error_count: int = 0
    success_count: int = 0
    last_error: Optional[str] = None
    uptime_percent: float = 100.0


@dataclass
class DataMapping:
    """Data field mapping configuration."""

    source_field: str
    target_field: str
    transform: Optional[str] = None  # e.g., "uppercase", "lowercase", "trim"
    default_value: Any = None
    required: bool = False


@dataclass
class TransformResult:
    """Data transformation result."""

    success: bool
    data: Any = None
    source_format: DataFormat = DataFormat.JSON
    target_format: DataFormat = DataFormat.JSON
    errors: List[str] = field(default_factory=list)


# ========================
# Connector Base
# ========================


class Connector(ABC):
    """Abstract base class for connectors."""

    def __init__(self, config: ConnectorConfig):
        """Initialize connector."""
        self.config = config
        self._status = ConnectorStatus.INACTIVE
        self._health = IntegrationHealth(
            connector_id=config.connector_id,
            status=ConnectorStatus.INACTIVE,
            last_check=datetime.now(),
        )

    @property
    def connector_id(self) -> str:
        """Get connector ID."""
        return self.config.connector_id

    @property
    def status(self) -> ConnectorStatus:
        """Get connector status."""
        return self._status

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection."""
        pass

    @abstractmethod
    async def execute(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an operation."""
        pass

    @abstractmethod
    async def health_check(self) -> IntegrationHealth:
        """Check connector health."""
        pass


class RESTConnector(Connector):
    """REST API connector."""

    def __init__(self, config: ConnectorConfig):
        """Initialize REST connector."""
        super().__init__(config)
        self._session: Optional[Any] = None
        self._token_cache: Optional[str] = None
        self._token_expires: Optional[datetime] = None

    async def connect(self) -> bool:
        """Establish REST connection."""
        try:
            self._status = ConnectorStatus.CONNECTING
            # In a real implementation, this would create an HTTP client
            self._status = ConnectorStatus.ACTIVE
            self._health.status = ConnectorStatus.ACTIVE
            self._health.last_check = datetime.now()
            return True
        except Exception as e:
            self._status = ConnectorStatus.ERROR
            self._health.last_error = str(e)
            return False

    async def disconnect(self) -> bool:
        """Close REST connection."""
        try:
            self._session = None
            self._status = ConnectorStatus.INACTIVE
            self._health.status = ConnectorStatus.INACTIVE
            return True
        except Exception:
            return False

    async def execute(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute REST API call."""
        start_time = time.time()

        try:
            method = params.get("method", "GET")
            endpoint = params.get("endpoint", "/")
            body = params.get("body")
            query_params = params.get("query", {})

            # Build URL
            url = f"{self.config.base_url.rstrip('/')}{endpoint}"

            # Get headers with auth
            headers = {**self.config.headers}
            if self.config.auth:
                auth_header = self._get_auth_header()
                if auth_header:
                    headers.update(auth_header)

            # Simulate API call result
            result = {
                "status": "success",
                "operation": operation,
                "method": method,
                "url": url,
                "response_time_ms": (time.time() - start_time) * 1000,
            }

            self._health.success_count += 1
            self._health.response_time_ms = result["response_time_ms"]

            return result

        except Exception as e:
            self._health.error_count += 1
            self._health.last_error = str(e)
            raise

    async def health_check(self) -> IntegrationHealth:
        """Check REST API health."""
        start_time = time.time()

        try:
            # Simulate health check
            self._health.response_time_ms = (time.time() - start_time) * 1000
            self._health.status = ConnectorStatus.ACTIVE
            self._health.last_check = datetime.now()

            # Calculate uptime
            total = self._health.success_count + self._health.error_count
            if total > 0:
                self._health.uptime_percent = (self._health.success_count / total) * 100

        except Exception as e:
            self._health.status = ConnectorStatus.ERROR
            self._health.last_error = str(e)

        return self._health

    def _get_auth_header(self) -> Dict[str, str]:
        """Get authentication header."""
        if not self.config.auth:
            return {}

        auth = self.config.auth

        if auth.auth_type == AuthenticationType.API_KEY:
            return {"X-API-Key": auth.api_key or ""}

        if auth.auth_type == AuthenticationType.BEARER:
            return {"Authorization": f"Bearer {auth.token or ''}"}

        if auth.auth_type == AuthenticationType.BASIC:
            import base64

            credentials = base64.b64encode(f"{auth.username}:{auth.password}".encode()).decode()
            return {"Authorization": f"Basic {credentials}"}

        return auth.custom_headers


class WebSocketConnector(Connector):
    """WebSocket connector."""

    def __init__(self, config: ConnectorConfig):
        """Initialize WebSocket connector."""
        super().__init__(config)
        self._connection: Optional[Any] = None
        self._message_handlers: List[Callable] = []

    async def connect(self) -> bool:
        """Establish WebSocket connection."""
        try:
            self._status = ConnectorStatus.CONNECTING
            # Simulate WebSocket connection
            self._status = ConnectorStatus.ACTIVE
            self._health.status = ConnectorStatus.ACTIVE
            return True
        except Exception as e:
            self._status = ConnectorStatus.ERROR
            self._health.last_error = str(e)
            return False

    async def disconnect(self) -> bool:
        """Close WebSocket connection."""
        self._connection = None
        self._status = ConnectorStatus.INACTIVE
        return True

    async def execute(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send WebSocket message."""
        message = params.get("message", {})
        return {"status": "sent", "operation": operation, "message": message}

    async def health_check(self) -> IntegrationHealth:
        """Check WebSocket health."""
        self._health.last_check = datetime.now()
        if self._status == ConnectorStatus.ACTIVE:
            self._health.status = ConnectorStatus.ACTIVE
        return self._health

    def on_message(self, handler: Callable) -> None:
        """Register message handler."""
        self._message_handlers.append(handler)


# ========================
# Webhook Manager
# ========================


class WebhookManager:
    """Manages webhook registrations and deliveries."""

    def __init__(self):
        """Initialize webhook manager."""
        self._lock = threading.Lock()
        self._webhooks: Dict[str, WebhookConfig] = {}
        self._delivery_queue: List[Tuple[str, WebhookPayload]] = []
        self._delivery_history: List[Dict[str, Any]] = []

    def register_webhook(self, config: WebhookConfig) -> bool:
        """Register a webhook."""
        with self._lock:
            self._webhooks[config.webhook_id] = config
            return True

    def unregister_webhook(self, webhook_id: str) -> bool:
        """Unregister a webhook."""
        with self._lock:
            if webhook_id in self._webhooks:
                del self._webhooks[webhook_id]
                return True
            return False

    def get_webhook(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get webhook configuration."""
        with self._lock:
            return self._webhooks.get(webhook_id)

    def list_webhooks(self) -> List[WebhookConfig]:
        """List all webhooks."""
        with self._lock:
            return list(self._webhooks.values())

    async def dispatch_event(self, event_type: WebhookEventType, data: Dict[str, Any]) -> List[str]:
        """Dispatch event to all subscribed webhooks."""
        delivered_to = []

        with self._lock:
            matching_webhooks = [
                wh
                for wh in self._webhooks.values()
                if wh.enabled and (event_type in wh.events or WebhookEventType.CUSTOM in wh.events)
            ]

        for webhook in matching_webhooks:
            payload = WebhookPayload(
                event_type=event_type, data=data, webhook_id=webhook.webhook_id
            )

            if webhook.secret:
                payload.signature = self._sign_payload(payload, webhook.secret)

            success = await self._deliver_webhook(webhook, payload)
            if success:
                delivered_to.append(webhook.webhook_id)

        return delivered_to

    async def _deliver_webhook(self, webhook: WebhookConfig, payload: WebhookPayload) -> bool:
        """Deliver webhook payload."""
        for attempt in range(webhook.retry_count):
            try:
                # In a real implementation, this would make an HTTP POST
                delivery_record = {
                    "webhook_id": webhook.webhook_id,
                    "event_type": payload.event_type.value,
                    "url": webhook.url,
                    "timestamp": datetime.now().isoformat(),
                    "attempt": attempt + 1,
                    "success": True,
                }

                with self._lock:
                    self._delivery_history.append(delivery_record)

                return True

            except Exception as e:
                if attempt == webhook.retry_count - 1:
                    with self._lock:
                        self._delivery_history.append(
                            {
                                "webhook_id": webhook.webhook_id,
                                "event_type": payload.event_type.value,
                                "url": webhook.url,
                                "timestamp": datetime.now().isoformat(),
                                "attempt": attempt + 1,
                                "success": False,
                                "error": str(e),
                            }
                        )

                await asyncio.sleep(1.0 * (attempt + 1))

        return False

    def _sign_payload(self, payload: WebhookPayload, secret: str) -> str:
        """Sign webhook payload."""
        data = json.dumps(
            {
                "event_type": payload.event_type.value,
                "data": payload.data,
                "timestamp": payload.timestamp.isoformat(),
            }
        )
        signature = hmac.new(secret.encode(), data.encode(), hashlib.sha256).hexdigest()
        return f"sha256={signature}"

    def verify_signature(self, payload: str, signature: str, secret: str) -> bool:
        """Verify webhook signature."""
        expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(f"sha256={expected}", signature)

    def get_delivery_history(
        self, webhook_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get delivery history."""
        with self._lock:
            history = self._delivery_history
            if webhook_id:
                history = [h for h in history if h["webhook_id"] == webhook_id]
            return history[-limit:]


# ========================
# Rate Limiter
# ========================


class RateLimiter:
    """Rate limiter for external service calls."""

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter."""
        self.config = config
        self._lock = threading.Lock()
        self._request_times: List[float] = []
        self._blocked_until: Optional[datetime] = None

    def can_proceed(self) -> bool:
        """Check if request can proceed."""
        if not self.config.enabled:
            return True

        with self._lock:
            now = time.time()

            # Check if blocked
            if self._blocked_until:
                if datetime.now() < self._blocked_until:
                    return False
                self._blocked_until = None

            # Clean old entries
            cutoff_second = now - 1.0
            cutoff_minute = now - 60.0
            cutoff_hour = now - 3600.0

            self._request_times = [t for t in self._request_times if t > cutoff_hour]

            # Check limits
            recent_second = sum(1 for t in self._request_times if t > cutoff_second)
            recent_minute = sum(1 for t in self._request_times if t > cutoff_minute)
            recent_hour = len(self._request_times)

            if recent_second >= self.config.requests_per_second:
                return False
            if recent_minute >= self.config.requests_per_minute:
                return False
            if recent_hour >= self.config.requests_per_hour:
                return False

            return True

    def record_request(self) -> None:
        """Record a request."""
        with self._lock:
            self._request_times.append(time.time())

    def block(self, seconds: Optional[int] = None) -> None:
        """Block requests for a period."""
        duration = seconds or self.config.retry_after
        with self._lock:
            self._blocked_until = datetime.now() + timedelta(seconds=duration)

    def get_wait_time(self) -> float:
        """Get time to wait before next request."""
        if not self.config.enabled:
            return 0.0

        with self._lock:
            if self._blocked_until:
                remaining = (self._blocked_until - datetime.now()).total_seconds()
                return max(0.0, remaining)

            now = time.time()
            recent = [t for t in self._request_times if t > now - 1.0]

            if len(recent) >= self.config.requests_per_second:
                return 1.0 - (now - recent[0])

            return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            now = time.time()
            return {
                "requests_last_second": sum(1 for t in self._request_times if t > now - 1.0),
                "requests_last_minute": sum(1 for t in self._request_times if t > now - 60.0),
                "requests_last_hour": len([t for t in self._request_times if t > now - 3600.0]),
                "blocked": self._blocked_until is not None,
                "blocked_until": self._blocked_until.isoformat() if self._blocked_until else None,
            }


# ========================
# Data Transformer
# ========================


class DataTransformer:
    """Transforms data between formats and structures."""

    def __init__(self):
        """Initialize data transformer."""
        self._transforms: Dict[str, Callable[[Any], Any]] = {
            "uppercase": lambda x: x.upper() if isinstance(x, str) else x,
            "lowercase": lambda x: x.lower() if isinstance(x, str) else x,
            "trim": lambda x: x.strip() if isinstance(x, str) else x,
            "int": lambda x: int(x) if x is not None else None,
            "float": lambda x: float(x) if x is not None else None,
            "string": str,
            "bool": lambda x: bool(x) if x is not None else None,
            "json_parse": lambda x: json.loads(x) if isinstance(x, str) else x,
            "json_stringify": json.dumps,
        }

    def register_transform(self, name: str, transform: Callable[[Any], Any]) -> None:
        """Register a custom transform."""
        self._transforms[name] = transform

    def apply_mapping(self, data: Dict[str, Any], mappings: List[DataMapping]) -> Dict[str, Any]:
        """Apply field mappings to data."""
        result = {}

        for mapping in mappings:
            # Get source value
            value = self._get_nested_value(data, mapping.source_field)

            if value is None:
                if mapping.required:
                    raise ValueError(f"Required field missing: {mapping.source_field}")
                value = mapping.default_value

            # Apply transform
            if mapping.transform and value is not None:
                transform_fn = self._transforms.get(mapping.transform)
                if transform_fn:
                    value = transform_fn(value)

            # Set target value
            self._set_nested_value(result, mapping.target_field, value)

        return result

    def transform_format(
        self, data: Any, source_format: DataFormat, target_format: DataFormat
    ) -> TransformResult:
        """Transform data between formats."""
        try:
            # Parse source
            if source_format == DataFormat.JSON and isinstance(data, str):
                parsed = json.loads(data)
            elif source_format == DataFormat.XML:
                parsed = self._parse_xml(data)
            elif source_format == DataFormat.CSV:
                parsed = self._parse_csv(data)
            else:
                parsed = data

            # Convert to target
            if target_format == DataFormat.JSON:
                output = json.dumps(parsed, indent=2)
            elif target_format == DataFormat.XML:
                output = self._to_xml(parsed)
            elif target_format == DataFormat.CSV:
                output = self._to_csv(parsed)
            else:
                output = parsed

            return TransformResult(
                success=True, data=output, source_format=source_format, target_format=target_format
            )

        except Exception as e:
            return TransformResult(
                success=False,
                source_format=source_format,
                target_format=target_format,
                errors=[str(e)],
            )

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dict using dot notation."""
        keys = path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value

    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested value in dict using dot notation."""
        keys = path.split(".")
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def _parse_xml(self, data: str) -> Dict[str, Any]:
        """Parse XML to dict (simplified)."""
        # Simplified XML parsing
        result: Dict[str, Any] = {}
        # Extract simple key-value pairs
        pattern = r"<(\w+)>([^<]+)</\1>"
        for match in re.finditer(pattern, data):
            result[match.group(1)] = match.group(2)
        return result

    def _to_xml(self, data: Dict[str, Any], root: str = "root") -> str:
        """Convert dict to XML (simplified)."""
        lines = [f"<{root}>"]
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(self._to_xml(value, key))
            else:
                lines.append(f"  <{key}>{value}</{key}>")
        lines.append(f"</{root}>")
        return "\n".join(lines)

    def _parse_csv(self, data: str) -> List[Dict[str, str]]:
        """Parse CSV to list of dicts."""
        lines = data.strip().split("\n")
        if not lines:
            return []

        headers = lines[0].split(",")
        result = []

        for line in lines[1:]:
            values = line.split(",")
            row = dict(zip(headers, values))
            result.append(row)

        return result

    def _to_csv(self, data: Any) -> str:
        """Convert to CSV."""
        if isinstance(data, list) and data and isinstance(data[0], dict):
            headers = list(data[0].keys())
            lines = [",".join(headers)]
            for row in data:
                values = [str(row.get(h, "")) for h in headers]
                lines.append(",".join(values))
            return "\n".join(lines)

        return str(data)


# ========================
# Integration Hub
# ========================


class IntegrationHub:
    """Central hub for managing integrations."""

    def __init__(self):
        """Initialize integration hub."""
        self._lock = threading.Lock()
        self._connectors: Dict[str, Connector] = {}
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._webhook_manager = WebhookManager()
        self._transformer = DataTransformer()
        self._event_handlers: Dict[str, List[Callable]] = {}

    def register_connector(self, connector: Connector) -> bool:
        """Register a connector."""
        with self._lock:
            self._connectors[connector.connector_id] = connector

            if connector.config.rate_limit:
                self._rate_limiters[connector.connector_id] = RateLimiter(
                    connector.config.rate_limit
                )

            return True

    def unregister_connector(self, connector_id: str) -> bool:
        """Unregister a connector."""
        with self._lock:
            if connector_id in self._connectors:
                del self._connectors[connector_id]
                if connector_id in self._rate_limiters:
                    del self._rate_limiters[connector_id]
                return True
            return False

    def get_connector(self, connector_id: str) -> Optional[Connector]:
        """Get a connector by ID."""
        with self._lock:
            return self._connectors.get(connector_id)

    def list_connectors(self) -> List[str]:
        """List all connector IDs."""
        with self._lock:
            return list(self._connectors.keys())

    async def connect_all(self) -> Dict[str, bool]:
        """Connect all connectors."""
        results = {}
        with self._lock:
            connectors = list(self._connectors.values())

        for connector in connectors:
            success = await connector.connect()
            results[connector.connector_id] = success

            if success:
                await self._webhook_manager.dispatch_event(
                    WebhookEventType.PROVIDER_CONNECTED, {"connector_id": connector.connector_id}
                )

        return results

    async def disconnect_all(self) -> Dict[str, bool]:
        """Disconnect all connectors."""
        results = {}
        with self._lock:
            connectors = list(self._connectors.values())

        for connector in connectors:
            success = await connector.disconnect()
            results[connector.connector_id] = success

            if success:
                await self._webhook_manager.dispatch_event(
                    WebhookEventType.PROVIDER_DISCONNECTED, {"connector_id": connector.connector_id}
                )

        return results

    async def execute(
        self, connector_id: str, operation: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute operation on a connector."""
        connector = self.get_connector(connector_id)
        if not connector:
            raise ValueError(f"Connector not found: {connector_id}")

        # Check rate limit
        rate_limiter = self._rate_limiters.get(connector_id)
        if rate_limiter and not rate_limiter.can_proceed():
            wait_time = rate_limiter.get_wait_time()
            await self._webhook_manager.dispatch_event(
                WebhookEventType.RATE_LIMIT_REACHED,
                {"connector_id": connector_id, "wait_time": wait_time},
            )
            raise RuntimeError(f"Rate limited. Wait {wait_time:.2f}s")

        try:
            result = await connector.execute(operation, params)

            if rate_limiter:
                rate_limiter.record_request()

            return result

        except Exception as e:
            await self._webhook_manager.dispatch_event(
                WebhookEventType.ERROR_OCCURRED,
                {"connector_id": connector_id, "operation": operation, "error": str(e)},
            )
            raise

    async def health_check_all(self) -> Dict[str, IntegrationHealth]:
        """Check health of all connectors."""
        results = {}
        with self._lock:
            connectors = list(self._connectors.values())

        for connector in connectors:
            health = await connector.health_check()
            results[connector.connector_id] = health

        await self._webhook_manager.dispatch_event(
            WebhookEventType.HEALTH_CHECK,
            {"connectors": {k: v.status.value for k, v in results.items()}},
        )

        return results

    # Webhook management
    def register_webhook(self, config: WebhookConfig) -> bool:
        """Register a webhook."""
        return self._webhook_manager.register_webhook(config)

    def unregister_webhook(self, webhook_id: str) -> bool:
        """Unregister a webhook."""
        return self._webhook_manager.unregister_webhook(webhook_id)

    async def dispatch_event(self, event_type: WebhookEventType, data: Dict[str, Any]) -> List[str]:
        """Dispatch webhook event."""
        return await self._webhook_manager.dispatch_event(event_type, data)

    # Data transformation
    def transform_data(self, data: Dict[str, Any], mappings: List[DataMapping]) -> Dict[str, Any]:
        """Transform data using mappings."""
        return self._transformer.apply_mapping(data, mappings)

    def convert_format(
        self, data: Any, source_format: DataFormat, target_format: DataFormat
    ) -> TransformResult:
        """Convert data format."""
        return self._transformer.transform_format(data, source_format, target_format)

    # Event handling
    def on_event(self, event_type: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register event handler."""
        with self._lock:
            if event_type not in self._event_handlers:
                self._event_handlers[event_type] = []
            self._event_handlers[event_type].append(handler)

    def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event to handlers."""
        with self._lock:
            handlers = self._event_handlers.get(event_type, [])

        for handler in handlers:
            try:
                handler(data)
            except Exception:
                pass  # Log in production


# ========================
# Vision Provider
# ========================


class IntegrationHubVisionProvider(VisionProvider):
    """Vision provider with integration hub capabilities."""

    def __init__(
        self, base_provider: VisionProvider, integration_hub: Optional[IntegrationHub] = None
    ):
        """Initialize integration hub vision provider."""
        self._base_provider = base_provider
        self._hub = integration_hub or IntegrationHub()

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"integration_hub_{self._base_provider.provider_name}"

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True, **kwargs: Any
    ) -> VisionDescription:
        """Analyze image with integration support."""
        # Dispatch analysis started event
        await self._hub.dispatch_event(
            WebhookEventType.ANALYSIS_STARTED, {"provider": self._base_provider.provider_name}
        )

        try:
            result = await self._base_provider.analyze_image(
                image_data, include_description=include_description, **kwargs
            )

            # Dispatch analysis completed event
            await self._hub.dispatch_event(
                WebhookEventType.ANALYSIS_COMPLETED,
                {
                    "provider": self._base_provider.provider_name,
                    "summary": result.summary,
                    "confidence": result.confidence,
                },
            )

            return result

        except Exception as e:
            # Dispatch analysis failed event
            await self._hub.dispatch_event(
                WebhookEventType.ANALYSIS_FAILED,
                {"provider": self._base_provider.provider_name, "error": str(e)},
            )
            raise

    def register_connector(self, connector: Connector) -> bool:
        """Register a connector."""
        return self._hub.register_connector(connector)

    async def execute_integration(
        self, connector_id: str, operation: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute integration operation."""
        return await self._hub.execute(connector_id, operation, params)

    def register_webhook(self, config: WebhookConfig) -> bool:
        """Register a webhook."""
        return self._hub.register_webhook(config)


# ========================
# Factory Functions
# ========================


def create_integration_hub() -> IntegrationHub:
    """Create a new integration hub instance."""
    return IntegrationHub()


def create_rest_connector(config: ConnectorConfig) -> RESTConnector:
    """Create a REST API connector."""
    return RESTConnector(config)


def create_websocket_connector(config: ConnectorConfig) -> WebSocketConnector:
    """Create a WebSocket connector."""
    return WebSocketConnector(config)


def create_webhook_manager() -> WebhookManager:
    """Create a webhook manager."""
    return WebhookManager()


def create_rate_limiter(config: RateLimitConfig) -> RateLimiter:
    """Create a rate limiter."""
    return RateLimiter(config)


def create_data_transformer() -> DataTransformer:
    """Create a data transformer."""
    return DataTransformer()


def create_integration_hub_provider(
    base_provider: VisionProvider, hub: Optional[IntegrationHub] = None
) -> IntegrationHubVisionProvider:
    """Create an integration hub vision provider."""
    return IntegrationHubVisionProvider(base_provider, hub)
