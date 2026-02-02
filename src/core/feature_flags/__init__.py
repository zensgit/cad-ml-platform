"""Feature Flags System for CAD ML Platform.

Provides runtime feature toggling with:
- Environment-based configuration
- Percentage rollouts
- User/tenant targeting
- A/B testing support
"""

from src.core.feature_flags.client import (
    FeatureFlag,
    FeatureFlagClient,
    FlagContext,
    RolloutStrategy,
    get_feature_client,
    is_enabled,
)

__all__ = [
    "FeatureFlag",
    "FeatureFlagClient",
    "FlagContext",
    "RolloutStrategy",
    "get_feature_client",
    "is_enabled",
]
