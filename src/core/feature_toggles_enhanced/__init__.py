"""Enhanced Feature Toggles Module.

Provides advanced feature toggle capabilities:
- Dynamic feature flags
- A/B testing
- Gradual rollout
- Multiple storage backends
"""

from src.core.feature_toggles_enhanced.toggle import (
    ToggleState,
    ToggleType,
    EvaluationContext,
    ToggleMetadata,
    EvaluationResult,
    ToggleRule,
    AlwaysOnRule,
    AlwaysOffRule,
    PercentageRule,
    UserIdRule,
    AttributeRule,
    TimeBasedRule,
    CompositeRule,
    FeatureToggle,
    ToggleListener,
    LoggingToggleListener,
)
from src.core.feature_toggles_enhanced.store import (
    ToggleStore,
    InMemoryToggleStore,
    FileToggleStore,
    CachingToggleStore,
)
from src.core.feature_toggles_enhanced.manager import (
    EvaluationMetrics,
    RolloutConfig,
    ToggleManager,
    ToggleClient,
    feature_flag,
)

__all__ = [
    # Toggle Core
    "ToggleState",
    "ToggleType",
    "EvaluationContext",
    "ToggleMetadata",
    "EvaluationResult",
    "ToggleRule",
    "AlwaysOnRule",
    "AlwaysOffRule",
    "PercentageRule",
    "UserIdRule",
    "AttributeRule",
    "TimeBasedRule",
    "CompositeRule",
    "FeatureToggle",
    "ToggleListener",
    "LoggingToggleListener",
    # Store
    "ToggleStore",
    "InMemoryToggleStore",
    "FileToggleStore",
    "CachingToggleStore",
    # Manager
    "EvaluationMetrics",
    "RolloutConfig",
    "ToggleManager",
    "ToggleClient",
    "feature_flag",
]
