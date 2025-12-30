"""Feature flag system for Vision Provider system.

This module provides feature flag capabilities including:
- Boolean and multivariate flags
- User/context-based targeting
- Percentage rollouts
- A/B testing integration
- Flag evaluation caching
"""

import hashlib
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Set, TypeVar, Union

from .base import VisionDescription, VisionProvider


class FlagType(Enum):
    """Type of feature flag."""

    BOOLEAN = "boolean"
    STRING = "string"
    NUMBER = "number"
    JSON = "json"


class FlagStatus(Enum):
    """Status of a feature flag."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


class TargetingOperator(Enum):
    """Operator for targeting rules."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    MATCHES = "matches"


@dataclass
class FlagVariation:
    """Variation of a feature flag."""

    variation_id: str
    value: Any
    name: str = ""
    description: str = ""
    weight: int = 0  # For percentage rollouts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variation_id": self.variation_id,
            "value": self.value,
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
        }


@dataclass
class TargetingRule:
    """Targeting rule for feature flag."""

    rule_id: str
    attribute: str
    operator: TargetingOperator
    values: List[Any]
    variation_id: str
    priority: int = 0

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate rule against context.

        Args:
            context: Evaluation context

        Returns:
            True if rule matches
        """
        ctx_value = context.get(self.attribute)
        if ctx_value is None:
            return False

        if self.operator == TargetingOperator.EQUALS:
            return ctx_value == self.values[0] if self.values else False

        elif self.operator == TargetingOperator.NOT_EQUALS:
            return ctx_value != self.values[0] if self.values else True

        elif self.operator == TargetingOperator.CONTAINS:
            if isinstance(ctx_value, str):
                return any(v in ctx_value for v in self.values)
            return False

        elif self.operator == TargetingOperator.NOT_CONTAINS:
            if isinstance(ctx_value, str):
                return not any(v in ctx_value for v in self.values)
            return True

        elif self.operator == TargetingOperator.IN:
            return ctx_value in self.values

        elif self.operator == TargetingOperator.NOT_IN:
            return ctx_value not in self.values

        elif self.operator == TargetingOperator.GREATER_THAN:
            try:
                return float(ctx_value) > float(self.values[0])
            except (ValueError, TypeError):
                return False

        elif self.operator == TargetingOperator.LESS_THAN:
            try:
                return float(ctx_value) < float(self.values[0])
            except (ValueError, TypeError):
                return False

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "attribute": self.attribute,
            "operator": self.operator.value,
            "values": list(self.values),
            "variation_id": self.variation_id,
            "priority": self.priority,
        }


@dataclass
class PercentageRollout:
    """Percentage-based rollout configuration."""

    rollout_id: str
    variations: List[FlagVariation]
    bucket_by: str = "user_id"

    def evaluate(self, context: Dict[str, Any]) -> Optional[FlagVariation]:
        """Evaluate rollout based on percentage.

        Args:
            context: Evaluation context

        Returns:
            Selected variation or None
        """
        bucket_value = context.get(self.bucket_by, "")
        if not bucket_value:
            bucket_value = str(random.random())

        # Generate consistent bucket
        hash_input = f"{self.rollout_id}:{bucket_value}"
        hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest()[:8], 16)
        bucket = hash_value % 100

        # Find variation based on weight
        current = 0
        for variation in self.variations:
            current += variation.weight
            if bucket < current:
                return variation

        return self.variations[-1] if self.variations else None


@dataclass
class FeatureFlag:
    """Feature flag definition."""

    flag_key: str
    flag_type: FlagType
    variations: List[FlagVariation]
    default_variation_id: str
    status: FlagStatus = FlagStatus.ACTIVE
    name: str = ""
    description: str = ""
    targeting_rules: List[TargetingRule] = field(default_factory=list)
    percentage_rollout: Optional[PercentageRollout] = None
    prerequisites: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def get_variation(self, variation_id: str) -> Optional[FlagVariation]:
        """Get variation by ID.

        Args:
            variation_id: Variation ID

        Returns:
            Variation or None
        """
        for v in self.variations:
            if v.variation_id == variation_id:
                return v
        return None

    def get_default_variation(self) -> Optional[FlagVariation]:
        """Get default variation.

        Returns:
            Default variation or None
        """
        return self.get_variation(self.default_variation_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "flag_key": self.flag_key,
            "flag_type": self.flag_type.value,
            "variations": [v.to_dict() for v in self.variations],
            "default_variation_id": self.default_variation_id,
            "status": self.status.value,
            "name": self.name,
            "description": self.description,
            "targeting_rules": [r.to_dict() for r in self.targeting_rules],
            "prerequisites": list(self.prerequisites),
            "tags": list(self.tags),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class EvaluationContext:
    """Context for flag evaluation."""

    user_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for rule evaluation."""
        result = dict(self.attributes)
        result.update(self.custom)
        if self.user_id:
            result["user_id"] = self.user_id
        return result


@dataclass
class EvaluationResult:
    """Result of flag evaluation."""

    flag_key: str
    value: Any
    variation_id: str
    reason: str
    matched_rule_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "flag_key": self.flag_key,
            "value": self.value,
            "variation_id": self.variation_id,
            "reason": self.reason,
            "matched_rule_id": self.matched_rule_id,
            "timestamp": self.timestamp.isoformat(),
        }


class FlagStore(ABC):
    """Abstract base class for flag storage."""

    @abstractmethod
    def get_flag(self, flag_key: str) -> Optional[FeatureFlag]:
        """Get flag by key.

        Args:
            flag_key: Flag key

        Returns:
            Feature flag or None
        """
        pass

    @abstractmethod
    def get_all_flags(self) -> Dict[str, FeatureFlag]:
        """Get all flags.

        Returns:
            Dictionary of all flags
        """
        pass

    @abstractmethod
    def save_flag(self, flag: FeatureFlag) -> None:
        """Save flag.

        Args:
            flag: Feature flag
        """
        pass

    @abstractmethod
    def delete_flag(self, flag_key: str) -> bool:
        """Delete flag.

        Args:
            flag_key: Flag key

        Returns:
            True if deleted
        """
        pass


class InMemoryFlagStore(FlagStore):
    """In-memory flag store."""

    def __init__(self) -> None:
        """Initialize store."""
        self._flags: Dict[str, FeatureFlag] = {}
        self._lock = threading.Lock()

    def get_flag(self, flag_key: str) -> Optional[FeatureFlag]:
        """Get flag by key."""
        with self._lock:
            return self._flags.get(flag_key)

    def get_all_flags(self) -> Dict[str, FeatureFlag]:
        """Get all flags."""
        with self._lock:
            return dict(self._flags)

    def save_flag(self, flag: FeatureFlag) -> None:
        """Save flag."""
        with self._lock:
            flag.updated_at = datetime.now()
            self._flags[flag.flag_key] = flag

    def delete_flag(self, flag_key: str) -> bool:
        """Delete flag."""
        with self._lock:
            if flag_key in self._flags:
                del self._flags[flag_key]
                return True
            return False


@dataclass
class FlagEvaluationCache:
    """Cache for flag evaluations."""

    max_size: int = 1000
    ttl_seconds: float = 60.0

    _cache: Dict[str, tuple] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, cache_key: str) -> Optional[EvaluationResult]:
        """Get cached evaluation.

        Args:
            cache_key: Cache key

        Returns:
            Cached result or None
        """
        with self._lock:
            entry = self._cache.get(cache_key)
            if entry:
                result, timestamp = entry
                if time.time() - timestamp < self.ttl_seconds:
                    return result
                else:
                    del self._cache[cache_key]
            return None

    def set(self, cache_key: str, result: EvaluationResult) -> None:
        """Cache evaluation result.

        Args:
            cache_key: Cache key
            result: Evaluation result
        """
        with self._lock:
            if len(self._cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[cache_key] = (result, time.time())

    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()

    def generate_key(
        self,
        flag_key: str,
        context: EvaluationContext,
    ) -> str:
        """Generate cache key.

        Args:
            flag_key: Flag key
            context: Evaluation context

        Returns:
            Cache key
        """
        ctx_str = str(sorted(context.to_dict().items()))
        return f"{flag_key}:{hashlib.sha256(ctx_str.encode()).hexdigest()}"


class FlagEvaluator:
    """Evaluator for feature flags."""

    def __init__(
        self,
        store: FlagStore,
        cache: Optional[FlagEvaluationCache] = None,
    ) -> None:
        """Initialize evaluator.

        Args:
            store: Flag store
            cache: Optional evaluation cache
        """
        self._store = store
        self._cache = cache
        self._hooks: List[Callable[[str, EvaluationResult], None]] = []

    def add_hook(
        self,
        hook: Callable[[str, EvaluationResult], None],
    ) -> None:
        """Add evaluation hook.

        Args:
            hook: Hook function
        """
        self._hooks.append(hook)

    def evaluate(
        self,
        flag_key: str,
        context: Optional[EvaluationContext] = None,
        default_value: Any = None,
    ) -> EvaluationResult:
        """Evaluate feature flag.

        Args:
            flag_key: Flag key
            context: Evaluation context
            default_value: Default value if flag not found

        Returns:
            Evaluation result
        """
        context = context or EvaluationContext()

        # Check cache
        if self._cache:
            cache_key = self._cache.generate_key(flag_key, context)
            cached = self._cache.get(cache_key)
            if cached:
                return cached

        # Get flag
        flag = self._store.get_flag(flag_key)
        if not flag:
            result = EvaluationResult(
                flag_key=flag_key,
                value=default_value,
                variation_id="default",
                reason="FLAG_NOT_FOUND",
            )
            self._notify_hooks(flag_key, result)
            return result

        # Check status
        if flag.status != FlagStatus.ACTIVE:
            default_var = flag.get_default_variation()
            result = EvaluationResult(
                flag_key=flag_key,
                value=default_var.value if default_var else default_value,
                variation_id=flag.default_variation_id,
                reason="FLAG_INACTIVE",
            )
            self._notify_hooks(flag_key, result)
            return result

        # Check prerequisites
        for prereq_key in flag.prerequisites:
            prereq_result = self.evaluate(prereq_key, context)
            if not prereq_result.value:
                default_var = flag.get_default_variation()
                result = EvaluationResult(
                    flag_key=flag_key,
                    value=default_var.value if default_var else default_value,
                    variation_id=flag.default_variation_id,
                    reason="PREREQUISITE_FAILED",
                )
                self._notify_hooks(flag_key, result)
                return result

        # Evaluate targeting rules
        ctx_dict = context.to_dict()
        sorted_rules = sorted(flag.targeting_rules, key=lambda r: r.priority, reverse=True)

        for rule in sorted_rules:
            if rule.evaluate(ctx_dict):
                variation = flag.get_variation(rule.variation_id)
                if variation:
                    result = EvaluationResult(
                        flag_key=flag_key,
                        value=variation.value,
                        variation_id=variation.variation_id,
                        reason="TARGETING_RULE_MATCH",
                        matched_rule_id=rule.rule_id,
                    )

                    if self._cache:
                        self._cache.set(cache_key, result)

                    self._notify_hooks(flag_key, result)
                    return result

        # Evaluate percentage rollout
        if flag.percentage_rollout:
            variation = flag.percentage_rollout.evaluate(ctx_dict)
            if variation:
                result = EvaluationResult(
                    flag_key=flag_key,
                    value=variation.value,
                    variation_id=variation.variation_id,
                    reason="PERCENTAGE_ROLLOUT",
                )

                if self._cache:
                    self._cache.set(cache_key, result)

                self._notify_hooks(flag_key, result)
                return result

        # Return default
        default_var = flag.get_default_variation()
        result = EvaluationResult(
            flag_key=flag_key,
            value=default_var.value if default_var else default_value,
            variation_id=flag.default_variation_id,
            reason="DEFAULT",
        )

        if self._cache:
            self._cache.set(cache_key, result)

        self._notify_hooks(flag_key, result)
        return result

    def evaluate_bool(
        self,
        flag_key: str,
        context: Optional[EvaluationContext] = None,
        default_value: bool = False,
    ) -> bool:
        """Evaluate boolean flag.

        Args:
            flag_key: Flag key
            context: Evaluation context
            default_value: Default value

        Returns:
            Boolean value
        """
        result = self.evaluate(flag_key, context, default_value)
        return bool(result.value)

    def evaluate_string(
        self,
        flag_key: str,
        context: Optional[EvaluationContext] = None,
        default_value: str = "",
    ) -> str:
        """Evaluate string flag.

        Args:
            flag_key: Flag key
            context: Evaluation context
            default_value: Default value

        Returns:
            String value
        """
        result = self.evaluate(flag_key, context, default_value)
        return str(result.value)

    def evaluate_number(
        self,
        flag_key: str,
        context: Optional[EvaluationContext] = None,
        default_value: Union[int, float] = 0,
    ) -> Union[int, float]:
        """Evaluate number flag.

        Args:
            flag_key: Flag key
            context: Evaluation context
            default_value: Default value

        Returns:
            Number value
        """
        result = self.evaluate(flag_key, context, default_value)
        try:
            return float(result.value)
        except (ValueError, TypeError):
            return default_value

    def _notify_hooks(self, flag_key: str, result: EvaluationResult) -> None:
        """Notify evaluation hooks."""
        for hook in self._hooks:
            try:
                hook(flag_key, result)
            except Exception:
                pass


class FeatureFlagManager:
    """Manager for feature flags."""

    def __init__(
        self,
        store: Optional[FlagStore] = None,
        enable_cache: bool = True,
    ) -> None:
        """Initialize manager.

        Args:
            store: Flag store
            enable_cache: Enable evaluation cache
        """
        self._store = store or InMemoryFlagStore()
        self._cache = FlagEvaluationCache() if enable_cache else None
        self._evaluator = FlagEvaluator(self._store, self._cache)

    @property
    def evaluator(self) -> FlagEvaluator:
        """Return flag evaluator."""
        return self._evaluator

    def create_flag(
        self,
        flag_key: str,
        flag_type: FlagType,
        variations: List[FlagVariation],
        default_variation_id: str,
        **kwargs: Any,
    ) -> FeatureFlag:
        """Create a new feature flag.

        Args:
            flag_key: Flag key
            flag_type: Flag type
            variations: Flag variations
            default_variation_id: Default variation ID
            **kwargs: Additional flag properties

        Returns:
            Created flag
        """
        flag = FeatureFlag(
            flag_key=flag_key,
            flag_type=flag_type,
            variations=variations,
            default_variation_id=default_variation_id,
            **kwargs,
        )
        self._store.save_flag(flag)

        if self._cache:
            self._cache.clear()

        return flag

    def create_boolean_flag(
        self,
        flag_key: str,
        default_value: bool = False,
        **kwargs: Any,
    ) -> FeatureFlag:
        """Create a boolean feature flag.

        Args:
            flag_key: Flag key
            default_value: Default value
            **kwargs: Additional flag properties

        Returns:
            Created flag
        """
        variations = [
            FlagVariation(variation_id="true", value=True, name="Enabled"),
            FlagVariation(variation_id="false", value=False, name="Disabled"),
        ]

        return self.create_flag(
            flag_key=flag_key,
            flag_type=FlagType.BOOLEAN,
            variations=variations,
            default_variation_id="true" if default_value else "false",
            **kwargs,
        )

    def get_flag(self, flag_key: str) -> Optional[FeatureFlag]:
        """Get flag by key.

        Args:
            flag_key: Flag key

        Returns:
            Feature flag or None
        """
        return self._store.get_flag(flag_key)

    def update_flag(self, flag: FeatureFlag) -> None:
        """Update feature flag.

        Args:
            flag: Feature flag
        """
        self._store.save_flag(flag)

        if self._cache:
            self._cache.clear()

    def delete_flag(self, flag_key: str) -> bool:
        """Delete feature flag.

        Args:
            flag_key: Flag key

        Returns:
            True if deleted
        """
        result = self._store.delete_flag(flag_key)

        if result and self._cache:
            self._cache.clear()

        return result

    def list_flags(self) -> List[str]:
        """List all flag keys.

        Returns:
            List of flag keys
        """
        return list(self._store.get_all_flags().keys())

    def is_enabled(
        self,
        flag_key: str,
        context: Optional[EvaluationContext] = None,
    ) -> bool:
        """Check if flag is enabled.

        Args:
            flag_key: Flag key
            context: Evaluation context

        Returns:
            True if enabled
        """
        return self._evaluator.evaluate_bool(flag_key, context, False)

    def get_variation(
        self,
        flag_key: str,
        context: Optional[EvaluationContext] = None,
        default_value: Any = None,
    ) -> Any:
        """Get flag variation value.

        Args:
            flag_key: Flag key
            context: Evaluation context
            default_value: Default value

        Returns:
            Variation value
        """
        result = self._evaluator.evaluate(flag_key, context, default_value)
        return result.value


class FeatureFlagVisionProvider(VisionProvider):
    """Vision provider with feature flag support."""

    def __init__(
        self,
        provider: VisionProvider,
        flag_manager: FeatureFlagManager,
        feature_flag_key: str = "vision.enabled",
    ) -> None:
        """Initialize feature flag provider.

        Args:
            provider: Underlying vision provider
            flag_manager: Feature flag manager
            feature_flag_key: Flag key for enabling provider
        """
        self._provider = provider
        self._flag_manager = flag_manager
        self._feature_flag_key = feature_flag_key

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"feature_flag_{self._provider.provider_name}"

    @property
    def flag_manager(self) -> FeatureFlagManager:
        """Return flag manager."""
        return self._flag_manager

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
        context: Optional[EvaluationContext] = None,
    ) -> VisionDescription:
        """Analyze image with feature flag check.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description
            context: Optional evaluation context

        Returns:
            Vision analysis description
        """
        # Check if provider is enabled
        is_enabled = self._flag_manager.is_enabled(
            self._feature_flag_key,
            context,
        )

        if not is_enabled:
            raise RuntimeError(
                f"Vision provider disabled by feature flag: {self._feature_flag_key}"
            )

        return await self._provider.analyze_image(image_data, include_description)


def create_feature_flag_provider(
    provider: VisionProvider,
    flag_manager: Optional[FeatureFlagManager] = None,
    feature_flag_key: str = "vision.enabled",
    default_enabled: bool = True,
) -> FeatureFlagVisionProvider:
    """Create a feature flag vision provider.

    Args:
        provider: Underlying vision provider
        flag_manager: Optional flag manager
        feature_flag_key: Flag key for enabling
        default_enabled: Default enabled state

    Returns:
        FeatureFlagVisionProvider instance
    """
    if flag_manager is None:
        flag_manager = FeatureFlagManager()

    # Create default flag if not exists
    if not flag_manager.get_flag(feature_flag_key):
        flag_manager.create_boolean_flag(
            flag_key=feature_flag_key,
            default_value=default_enabled,
            name="Vision Provider Enabled",
            description="Controls whether vision provider is enabled",
        )

    return FeatureFlagVisionProvider(
        provider=provider,
        flag_manager=flag_manager,
        feature_flag_key=feature_flag_key,
    )
