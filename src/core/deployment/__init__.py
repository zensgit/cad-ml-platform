"""Deployment Management for CAD ML Platform.

Provides:
- Blue-Green deployment support
- A/B testing infrastructure
- Canary release management
"""

from src.core.deployment.blue_green import (
    BlueGreenManager,
    DeploymentSlot,
    SlotStatus,
)
from src.core.deployment.ab_testing import (
    ABTestManager,
    Experiment,
    ExperimentStatus,
    Variant,
)
from src.core.deployment.canary import (
    CanaryManager,
    CanaryConfig,
    CanaryStatus,
    RolloutPhase,
)

__all__ = [
    # Blue-Green
    "BlueGreenManager",
    "DeploymentSlot",
    "SlotStatus",
    # A/B Testing
    "ABTestManager",
    "Experiment",
    "ExperimentStatus",
    "Variant",
    # Canary
    "CanaryManager",
    "CanaryConfig",
    "CanaryStatus",
    "RolloutPhase",
]
