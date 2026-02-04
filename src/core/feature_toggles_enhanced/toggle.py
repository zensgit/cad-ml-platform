"""Feature Toggle Core.

Provides feature toggle functionality:
- Toggle definitions
- Evaluation context
- Toggle states
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class ToggleState(Enum):
    """Feature toggle state."""
    ON = "on"
    OFF = "off"
    CONDITIONAL = "conditional"


class ToggleType(Enum):
    """Type of feature toggle."""
    RELEASE = "release"       # Gradual rollout of new features
    EXPERIMENT = "experiment" # A/B testing
    OPS = "ops"              # Operational switches
    PERMISSION = "permission" # Access control


@dataclass
class EvaluationContext:
    """Context for evaluating feature toggles."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute value."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.attributes.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "country": self.country,
            "region": self.region,
            "attributes": self.attributes,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ToggleMetadata:
    """Metadata for a feature toggle."""
    name: str
    description: str = ""
    toggle_type: ToggleType = ToggleType.RELEASE
    owner: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: Set[str] = field(default_factory=set)
    expires_at: Optional[datetime] = None
    stale_days: int = 30  # Days before toggle is considered stale


@dataclass
class EvaluationResult:
    """Result of toggle evaluation."""
    enabled: bool
    toggle_name: str
    variant: Optional[str] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToggleRule(ABC):
    """Abstract base class for toggle rules."""

    @abstractmethod
    def evaluate(self, context: EvaluationContext) -> Optional[bool]:
        """Evaluate rule against context.

        Returns:
            True/False if rule matches, None if rule doesn't apply.
        """
        pass


class AlwaysOnRule(ToggleRule):
    """Rule that always returns True."""

    def evaluate(self, context: EvaluationContext) -> Optional[bool]:
        return True


class AlwaysOffRule(ToggleRule):
    """Rule that always returns False."""

    def evaluate(self, context: EvaluationContext) -> Optional[bool]:
        return False


class PercentageRule(ToggleRule):
    """Rule based on percentage rollout."""

    def __init__(
        self,
        percentage: float,
        sticky: bool = True,
        sticky_key: str = "user_id",
    ):
        if not 0 <= percentage <= 100:
            raise ValueError("Percentage must be between 0 and 100")
        self.percentage = percentage
        self.sticky = sticky
        self.sticky_key = sticky_key

    def evaluate(self, context: EvaluationContext) -> Optional[bool]:
        if self.sticky:
            # Use consistent hashing for sticky sessions
            key = context.get(self.sticky_key)
            if key:
                hash_val = int(hashlib.md5(str(key).encode()).hexdigest(), 16)  # nosec B324 - stable rollout hashing
                return (hash_val % 100) < self.percentage

        # Random evaluation
        return random.random() * 100 < self.percentage


class UserIdRule(ToggleRule):
    """Rule based on user ID list."""

    def __init__(self, user_ids: Set[str], include: bool = True):
        self.user_ids = user_ids
        self.include = include

    def evaluate(self, context: EvaluationContext) -> Optional[bool]:
        if not context.user_id:
            return None

        in_list = context.user_id in self.user_ids
        return in_list if self.include else not in_list


class AttributeRule(ToggleRule):
    """Rule based on context attribute."""

    def __init__(
        self,
        attribute: str,
        operator: str,
        value: Any,
    ):
        self.attribute = attribute
        self.operator = operator
        self.value = value

    def evaluate(self, context: EvaluationContext) -> Optional[bool]:
        attr_value = context.get(self.attribute)
        if attr_value is None:
            return None

        if self.operator == "eq":
            return attr_value == self.value
        elif self.operator == "ne":
            return attr_value != self.value
        elif self.operator == "gt":
            return attr_value > self.value
        elif self.operator == "gte":
            return attr_value >= self.value
        elif self.operator == "lt":
            return attr_value < self.value
        elif self.operator == "lte":
            return attr_value <= self.value
        elif self.operator == "in":
            return attr_value in self.value
        elif self.operator == "not_in":
            return attr_value not in self.value
        elif self.operator == "contains":
            return self.value in attr_value
        elif self.operator == "regex":
            return bool(re.match(self.value, str(attr_value)))
        else:
            return None


class TimeBasedRule(ToggleRule):
    """Rule based on time window."""

    def __init__(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ):
        self.start_time = start_time
        self.end_time = end_time

    def evaluate(self, context: EvaluationContext) -> Optional[bool]:
        now = context.timestamp

        if self.start_time and now < self.start_time:
            return False
        if self.end_time and now > self.end_time:
            return False

        return True


class CompositeRule(ToggleRule):
    """Combine multiple rules with AND/OR logic."""

    def __init__(self, rules: List[ToggleRule], require_all: bool = True):
        self.rules = rules
        self.require_all = require_all

    def evaluate(self, context: EvaluationContext) -> Optional[bool]:
        results = []
        for rule in self.rules:
            result = rule.evaluate(context)
            if result is not None:
                results.append(result)

        if not results:
            return None

        if self.require_all:
            return all(results)
        else:
            return any(results)


@dataclass
class FeatureToggle:
    """A feature toggle definition."""
    name: str
    state: ToggleState = ToggleState.OFF
    rules: List[ToggleRule] = field(default_factory=list)
    metadata: ToggleMetadata = field(default_factory=lambda: ToggleMetadata(name=""))
    default_value: bool = False
    variants: Dict[str, int] = field(default_factory=dict)  # variant -> weight

    def __post_init__(self):
        if not self.metadata.name:
            self.metadata = ToggleMetadata(name=self.name)

    def is_enabled(self, context: Optional[EvaluationContext] = None) -> bool:
        """Check if toggle is enabled."""
        return self.evaluate(context).enabled

    def evaluate(self, context: Optional[EvaluationContext] = None) -> EvaluationResult:
        """Evaluate toggle for given context."""
        context = context or EvaluationContext()

        # Check expiration
        if self.metadata.expires_at and datetime.utcnow() > self.metadata.expires_at:
            return EvaluationResult(
                enabled=self.default_value,
                toggle_name=self.name,
                reason="toggle_expired",
            )

        # Check state
        if self.state == ToggleState.ON:
            return EvaluationResult(
                enabled=True,
                toggle_name=self.name,
                reason="toggle_on",
            )
        elif self.state == ToggleState.OFF:
            return EvaluationResult(
                enabled=False,
                toggle_name=self.name,
                reason="toggle_off",
            )

        # Evaluate rules
        for i, rule in enumerate(self.rules):
            result = rule.evaluate(context)
            if result is not None:
                return EvaluationResult(
                    enabled=result,
                    toggle_name=self.name,
                    reason=f"rule_{i}_{type(rule).__name__}",
                )

        # Default
        return EvaluationResult(
            enabled=self.default_value,
            toggle_name=self.name,
            reason="default_value",
        )

    def get_variant(self, context: Optional[EvaluationContext] = None) -> Optional[str]:
        """Get variant for A/B testing."""
        if not self.variants:
            return None

        context = context or EvaluationContext()

        # Use consistent hashing
        key = context.user_id or context.session_id or str(random.random())
        hash_val = int(hashlib.md5(f"{self.name}:{key}".encode()).hexdigest(), 16)  # nosec B324 - consistent variant bucketing

        total_weight = sum(self.variants.values())
        if total_weight == 0:
            return None

        threshold = hash_val % total_weight
        cumulative = 0

        for variant, weight in self.variants.items():
            cumulative += weight
            if threshold < cumulative:
                return variant

        return list(self.variants.keys())[-1]


class ToggleListener(ABC):
    """Listener for toggle changes."""

    @abstractmethod
    def on_toggle_changed(
        self,
        toggle_name: str,
        old_state: Optional[ToggleState],
        new_state: ToggleState,
    ) -> None:
        """Called when a toggle changes."""
        pass


class LoggingToggleListener(ToggleListener):
    """Listener that logs toggle changes."""

    def on_toggle_changed(
        self,
        toggle_name: str,
        old_state: Optional[ToggleState],
        new_state: ToggleState,
    ) -> None:
        logger.info(
            f"Toggle '{toggle_name}' changed: "
            f"{old_state.value if old_state else 'None'} -> {new_state.value}"
        )
