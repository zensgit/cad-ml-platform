"""Feature Flag Client Implementation.

Supports multiple backends:
- Environment variables (default)
- Redis (distributed)
- Config file (static)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class RolloutStrategy(Enum):
    """Feature flag rollout strategies."""

    ALL = "all"  # Enable for all users
    NONE = "none"  # Disable for all users
    PERCENTAGE = "percentage"  # Enable for percentage of users
    USER_LIST = "user_list"  # Enable for specific users
    TENANT_LIST = "tenant_list"  # Enable for specific tenants
    CUSTOM = "custom"  # Custom evaluation function


@dataclass
class FlagContext:
    """Context for evaluating feature flags."""

    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    environment: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def get_hash_key(self) -> str:
        """Get consistent hash key for percentage rollouts."""
        key = f"{self.user_id or ''}-{self.tenant_id or ''}"
        return hashlib.md5(key.encode()).hexdigest()  # nosec B324

    def get_percentage_bucket(self) -> int:
        """Get bucket (0-99) for percentage rollouts."""
        hash_key = self.get_hash_key()
        return int(hash_key[:8], 16) % 100


@dataclass
class FeatureFlag:
    """Feature flag definition."""

    name: str
    description: str = ""
    enabled: bool = False
    strategy: RolloutStrategy = RolloutStrategy.ALL
    percentage: int = 0  # For percentage rollout (0-100)
    allowed_users: Set[str] = field(default_factory=set)
    allowed_tenants: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def evaluate(self, context: Optional[FlagContext] = None) -> bool:
        """Evaluate flag for given context."""
        if not self.enabled:
            return False

        if self.strategy == RolloutStrategy.ALL:
            return True

        if self.strategy == RolloutStrategy.NONE:
            return False

        if context is None:
            return False

        if self.strategy == RolloutStrategy.PERCENTAGE:
            return context.get_percentage_bucket() < self.percentage

        if self.strategy == RolloutStrategy.USER_LIST:
            return context.user_id in self.allowed_users if context.user_id else False

        if self.strategy == RolloutStrategy.TENANT_LIST:
            return context.tenant_id in self.allowed_tenants if context.tenant_id else False

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "strategy": self.strategy.value,
            "percentage": self.percentage,
            "allowed_users": list(self.allowed_users),
            "allowed_tenants": list(self.allowed_tenants),
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureFlag":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            enabled=data.get("enabled", False),
            strategy=RolloutStrategy(data.get("strategy", "all")),
            percentage=data.get("percentage", 0),
            allowed_users=set(data.get("allowed_users", [])),
            allowed_tenants=set(data.get("allowed_tenants", [])),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )


class FeatureFlagClient:
    """Feature flag client with multiple backend support."""

    def __init__(
        self,
        backend: str = "env",
        redis_client: Optional[Any] = None,
        config_path: Optional[str] = None,
        prefix: str = "FF_",
        cache_ttl: int = 60,
    ):
        """Initialize feature flag client.

        Args:
            backend: Backend type ('env', 'redis', 'config')
            redis_client: Redis client for distributed flags
            config_path: Path to config file for static flags
            prefix: Environment variable prefix
            cache_ttl: Cache TTL in seconds
        """
        self.backend = backend
        self.redis_client = redis_client
        self.config_path = config_path
        self.prefix = prefix
        self.cache_ttl = cache_ttl

        self._flags: Dict[str, FeatureFlag] = {}
        self._cache: Dict[str, tuple] = {}  # (value, timestamp)
        self._custom_evaluators: Dict[str, Callable[[FlagContext], bool]] = {}

        self._load_flags()

    def _load_flags(self) -> None:
        """Load flags from backend."""
        if self.backend == "env":
            self._load_from_env()
        elif self.backend == "config":
            self._load_from_config()
        elif self.backend == "redis":
            self._load_from_redis()

    def _load_from_env(self) -> None:
        """Load flags from environment variables."""
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                flag_name = key[len(self.prefix) :].lower()
                enabled = value.lower() in ("1", "true", "yes", "on")

                # Check for percentage suffix
                percentage = 100 if enabled else 0
                strategy = RolloutStrategy.ALL if enabled else RolloutStrategy.NONE

                # Parse JSON config if present
                if value.startswith("{"):
                    try:
                        config = json.loads(value)
                        enabled = config.get("enabled", enabled)
                        percentage = config.get("percentage", percentage)
                        strategy = RolloutStrategy(config.get("strategy", strategy.value))
                    except json.JSONDecodeError:
                        pass

                self._flags[flag_name] = FeatureFlag(
                    name=flag_name,
                    enabled=enabled,
                    strategy=strategy,
                    percentage=percentage,
                )

    def _load_from_config(self) -> None:
        """Load flags from config file."""
        if not self.config_path or not os.path.exists(self.config_path):
            return

        try:
            with open(self.config_path) as f:
                config = json.load(f)
                for flag_data in config.get("flags", []):
                    flag = FeatureFlag.from_dict(flag_data)
                    self._flags[flag.name] = flag
        except Exception as e:
            logger.error(f"Failed to load feature flags from config: {e}")

    def _load_from_redis(self) -> None:
        """Load flags from Redis."""
        if not self.redis_client:
            return

        try:
            keys = self.redis_client.keys(f"{self.prefix}*")
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    flag_data = json.loads(data)
                    flag = FeatureFlag.from_dict(flag_data)
                    self._flags[flag.name] = flag
        except Exception as e:
            logger.error(f"Failed to load feature flags from Redis: {e}")

    def register_flag(self, flag: FeatureFlag) -> None:
        """Register a feature flag."""
        self._flags[flag.name] = flag
        logger.info(f"Registered feature flag: {flag.name}")

    def register_evaluator(
        self, flag_name: str, evaluator: Callable[[FlagContext], bool]
    ) -> None:
        """Register custom evaluator for a flag."""
        self._custom_evaluators[flag_name] = evaluator

    def is_enabled(
        self,
        flag_name: str,
        context: Optional[FlagContext] = None,
        default: bool = False,
    ) -> bool:
        """Check if a feature flag is enabled.

        Args:
            flag_name: Name of the feature flag
            context: Evaluation context
            default: Default value if flag not found

        Returns:
            True if flag is enabled
        """
        # Check cache
        cache_key = f"{flag_name}:{context.get_hash_key() if context else 'default'}"
        if cache_key in self._cache:
            value, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return value

        # Get flag
        flag = self._flags.get(flag_name)
        if flag is None:
            # Check environment as fallback
            env_key = f"{self.prefix}{flag_name.upper()}"
            env_value = os.getenv(env_key, "").lower()
            if env_value:
                result = env_value in ("1", "true", "yes", "on")
            else:
                result = default
        elif flag.strategy == RolloutStrategy.CUSTOM:
            evaluator = self._custom_evaluators.get(flag_name)
            result = evaluator(context) if evaluator and context else default
        else:
            result = flag.evaluate(context)

        # Update cache
        self._cache[cache_key] = (result, time.time())
        return result

    def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """Get a feature flag by name."""
        return self._flags.get(flag_name)

    def list_flags(self) -> List[FeatureFlag]:
        """List all registered flags."""
        return list(self._flags.values())

    def set_flag(
        self,
        flag_name: str,
        enabled: bool,
        strategy: Optional[RolloutStrategy] = None,
        percentage: Optional[int] = None,
    ) -> None:
        """Set flag state at runtime."""
        if flag_name in self._flags:
            flag = self._flags[flag_name]
            flag.enabled = enabled
            if strategy:
                flag.strategy = strategy
            if percentage is not None:
                flag.percentage = percentage
            flag.updated_at = time.time()
        else:
            self._flags[flag_name] = FeatureFlag(
                name=flag_name,
                enabled=enabled,
                strategy=strategy or RolloutStrategy.ALL,
                percentage=percentage or 0,
            )

        # Clear cache for this flag
        keys_to_remove = [k for k in self._cache if k.startswith(f"{flag_name}:")]
        for key in keys_to_remove:
            del self._cache[key]

        # Persist to Redis if using Redis backend
        if self.backend == "redis" and self.redis_client:
            try:
                self.redis_client.set(
                    f"{self.prefix}{flag_name}",
                    json.dumps(self._flags[flag_name].to_dict()),
                )
            except Exception as e:
                logger.error(f"Failed to persist flag to Redis: {e}")

    def clear_cache(self) -> None:
        """Clear the flag cache."""
        self._cache.clear()


# Global client instance
_client: Optional[FeatureFlagClient] = None


def get_feature_client() -> FeatureFlagClient:
    """Get global feature flag client."""
    global _client
    if _client is None:
        backend = os.getenv("FEATURE_FLAG_BACKEND", "env")
        config_path = os.getenv("FEATURE_FLAG_CONFIG_PATH")
        _client = FeatureFlagClient(backend=backend, config_path=config_path)
    return _client


def is_enabled(
    flag_name: str,
    context: Optional[FlagContext] = None,
    default: bool = False,
) -> bool:
    """Check if a feature flag is enabled (convenience function)."""
    return get_feature_client().is_enabled(flag_name, context, default)


# Pre-defined feature flags for CAD ML Platform
PREDEFINED_FLAGS = {
    "ocr_v2_enabled": FeatureFlag(
        name="ocr_v2_enabled",
        description="Enable OCR v2 with enhanced dimension extraction",
        enabled=False,
        strategy=RolloutStrategy.PERCENTAGE,
        percentage=0,
    ),
    "hybrid_classifier_enabled": FeatureFlag(
        name="hybrid_classifier_enabled",
        description="Enable hybrid classifier with 3-source fusion",
        enabled=False,
        strategy=RolloutStrategy.PERCENTAGE,
        percentage=0,
    ),
    "knowledge_graph_enabled": FeatureFlag(
        name="knowledge_graph_enabled",
        description="Enable knowledge graph for part recommendations",
        enabled=True,
        strategy=RolloutStrategy.ALL,
    ),
    "strict_rate_limiting": FeatureFlag(
        name="strict_rate_limiting",
        description="Enable strict rate limiting mode",
        enabled=False,
        strategy=RolloutStrategy.TENANT_LIST,
    ),
    "experimental_preprocessing": FeatureFlag(
        name="experimental_preprocessing",
        description="Enable experimental image preprocessing",
        enabled=False,
        strategy=RolloutStrategy.USER_LIST,
    ),
}


def register_predefined_flags() -> None:
    """Register pre-defined feature flags."""
    client = get_feature_client()
    for flag in PREDEFINED_FLAGS.values():
        if client.get_flag(flag.name) is None:
            client.register_flag(flag)
