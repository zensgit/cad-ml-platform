"""Feature Toggle Manager.

Provides high-level toggle management:
- Toggle evaluation
- A/B testing
- Gradual rollout
- Analytics
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

from src.core.feature_toggles_enhanced.toggle import (
    EvaluationContext,
    EvaluationResult,
    FeatureToggle,
    ToggleListener,
    ToggleMetadata,
    ToggleRule,
    ToggleState,
    ToggleType,
    PercentageRule,
)
from src.core.feature_toggles_enhanced.store import (
    InMemoryToggleStore,
    ToggleStore,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for toggle evaluations."""
    toggle_name: str
    total_evaluations: int = 0
    enabled_count: int = 0
    disabled_count: int = 0
    variant_counts: Dict[str, int] = field(default_factory=dict)
    last_evaluation: Optional[datetime] = None

    @property
    def enabled_rate(self) -> float:
        if self.total_evaluations == 0:
            return 0.0
        return self.enabled_count / self.total_evaluations


@dataclass
class RolloutConfig:
    """Configuration for gradual rollout."""
    initial_percentage: float = 0.0
    target_percentage: float = 100.0
    increment: float = 10.0
    interval_seconds: float = 3600.0  # 1 hour
    auto_rollback_on_error: bool = True
    error_threshold: float = 0.05  # 5% error rate triggers rollback


class ToggleManager:
    """High-level feature toggle management."""

    def __init__(
        self,
        store: Optional[ToggleStore] = None,
        default_context: Optional[EvaluationContext] = None,
    ):
        self.store = store or InMemoryToggleStore()
        self.default_context = default_context or EvaluationContext()
        self._metrics: Dict[str, EvaluationMetrics] = {}
        self._rollouts: Dict[str, RolloutConfig] = {}
        self._listeners: List[ToggleListener] = []

    def add_listener(self, listener: ToggleListener) -> None:
        """Add a toggle change listener."""
        self._listeners.append(listener)
        if isinstance(self.store, InMemoryToggleStore):
            self.store.add_listener(listener)

    async def is_enabled(
        self,
        name: str,
        context: Optional[EvaluationContext] = None,
    ) -> bool:
        """Check if a toggle is enabled."""
        result = await self.evaluate(name, context)
        return result.enabled

    async def evaluate(
        self,
        name: str,
        context: Optional[EvaluationContext] = None,
    ) -> EvaluationResult:
        """Evaluate a toggle."""
        toggle = await self.store.get(name)
        if not toggle:
            return EvaluationResult(
                enabled=False,
                toggle_name=name,
                reason="toggle_not_found",
            )

        ctx = context or self.default_context
        result = toggle.evaluate(ctx)

        # Track metrics
        self._record_evaluation(name, result)

        return result

    async def get_variant(
        self,
        name: str,
        context: Optional[EvaluationContext] = None,
    ) -> Optional[str]:
        """Get variant for A/B testing."""
        toggle = await self.store.get(name)
        if not toggle:
            return None

        ctx = context or self.default_context
        variant = toggle.get_variant(ctx)

        # Track variant
        if variant and name in self._metrics:
            counts = self._metrics[name].variant_counts
            counts[variant] = counts.get(variant, 0) + 1

        return variant

    def _record_evaluation(self, name: str, result: EvaluationResult) -> None:
        """Record evaluation metrics."""
        if name not in self._metrics:
            self._metrics[name] = EvaluationMetrics(toggle_name=name)

        metrics = self._metrics[name]
        metrics.total_evaluations += 1
        if result.enabled:
            metrics.enabled_count += 1
        else:
            metrics.disabled_count += 1
        metrics.last_evaluation = datetime.utcnow()

    async def create_toggle(
        self,
        name: str,
        state: ToggleState = ToggleState.OFF,
        toggle_type: ToggleType = ToggleType.RELEASE,
        description: str = "",
        owner: str = "",
        rules: Optional[List[ToggleRule]] = None,
        variants: Optional[Dict[str, int]] = None,
    ) -> FeatureToggle:
        """Create a new toggle."""
        toggle = FeatureToggle(
            name=name,
            state=state,
            rules=rules or [],
            variants=variants or {},
            metadata=ToggleMetadata(
                name=name,
                description=description,
                toggle_type=toggle_type,
                owner=owner,
            ),
        )
        await self.store.save(toggle)
        logger.info(f"Created toggle: {name}")
        return toggle

    async def update_state(self, name: str, state: ToggleState) -> bool:
        """Update toggle state."""
        toggle = await self.store.get(name)
        if not toggle:
            return False

        old_state = toggle.state
        toggle.state = state
        await self.store.save(toggle)

        logger.info(f"Toggle '{name}' state: {old_state.value} -> {state.value}")
        return True

    async def enable(self, name: str) -> bool:
        """Enable a toggle."""
        return await self.update_state(name, ToggleState.ON)

    async def disable(self, name: str) -> bool:
        """Disable a toggle."""
        return await self.update_state(name, ToggleState.OFF)

    async def delete_toggle(self, name: str) -> bool:
        """Delete a toggle."""
        result = await self.store.delete(name)
        if result:
            self._metrics.pop(name, None)
            self._rollouts.pop(name, None)
            logger.info(f"Deleted toggle: {name}")
        return result

    async def get_all_toggles(self) -> List[FeatureToggle]:
        """Get all toggles."""
        return await self.store.get_all()

    async def get_toggles_by_type(self, toggle_type: ToggleType) -> List[FeatureToggle]:
        """Get toggles by type."""
        all_toggles = await self.store.get_all()
        return [t for t in all_toggles if t.metadata.toggle_type == toggle_type]

    async def get_stale_toggles(self, days: int = 30) -> List[FeatureToggle]:
        """Get toggles that haven't been updated in a while."""
        all_toggles = await self.store.get_all()
        cutoff = datetime.utcnow() - timedelta(days=days)
        return [t for t in all_toggles if t.metadata.updated_at < cutoff]

    def get_metrics(self, name: str) -> Optional[EvaluationMetrics]:
        """Get evaluation metrics for a toggle."""
        return self._metrics.get(name)

    def get_all_metrics(self) -> Dict[str, EvaluationMetrics]:
        """Get all evaluation metrics."""
        return self._metrics.copy()

    # Gradual Rollout

    async def start_gradual_rollout(
        self,
        name: str,
        config: RolloutConfig,
    ) -> bool:
        """Start gradual rollout for a toggle."""
        toggle = await self.store.get(name)
        if not toggle:
            return False

        # Set initial percentage
        toggle.state = ToggleState.CONDITIONAL
        toggle.rules = [
            PercentageRule(
                percentage=config.initial_percentage,
                sticky=True,
            )
        ]
        await self.store.save(toggle)

        self._rollouts[name] = config
        logger.info(
            f"Started gradual rollout for '{name}': "
            f"{config.initial_percentage}% -> {config.target_percentage}%"
        )
        return True

    async def advance_rollout(self, name: str) -> Optional[float]:
        """Advance rollout to next percentage."""
        if name not in self._rollouts:
            return None

        toggle = await self.store.get(name)
        if not toggle:
            return None

        config = self._rollouts[name]

        # Find current percentage rule
        current_pct = 0.0
        for rule in toggle.rules:
            if isinstance(rule, PercentageRule):
                current_pct = rule.percentage
                break

        # Calculate new percentage
        new_pct = min(current_pct + config.increment, config.target_percentage)

        # Update rule
        toggle.rules = [
            PercentageRule(percentage=new_pct, sticky=True)
        ]
        await self.store.save(toggle)

        logger.info(f"Advanced rollout for '{name}': {current_pct}% -> {new_pct}%")

        # Check if complete
        if new_pct >= config.target_percentage:
            del self._rollouts[name]
            toggle.state = ToggleState.ON
            toggle.rules = []
            await self.store.save(toggle)
            logger.info(f"Completed rollout for '{name}'")

        return new_pct

    async def rollback(self, name: str) -> bool:
        """Rollback a toggle to disabled."""
        toggle = await self.store.get(name)
        if not toggle:
            return False

        toggle.state = ToggleState.OFF
        toggle.rules = []
        await self.store.save(toggle)

        self._rollouts.pop(name, None)
        logger.warning(f"Rolled back toggle: {name}")
        return True

    # A/B Testing

    async def create_experiment(
        self,
        name: str,
        variants: Dict[str, int],
        description: str = "",
        owner: str = "",
    ) -> FeatureToggle:
        """Create an A/B test experiment."""
        return await self.create_toggle(
            name=name,
            state=ToggleState.CONDITIONAL,
            toggle_type=ToggleType.EXPERIMENT,
            description=description,
            owner=owner,
            variants=variants,
        )

    def get_experiment_results(self, name: str) -> Optional[Dict[str, Any]]:
        """Get A/B test results."""
        metrics = self._metrics.get(name)
        if not metrics:
            return None

        return {
            "toggle_name": name,
            "total_evaluations": metrics.total_evaluations,
            "variant_distribution": metrics.variant_counts,
            "enabled_rate": metrics.enabled_rate,
        }


class ToggleClient:
    """Client for feature toggle evaluation with caching."""

    def __init__(
        self,
        manager: ToggleManager,
        cache_ttl: float = 10.0,
    ):
        self.manager = manager
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple[bool, float]] = {}

    async def is_enabled(
        self,
        name: str,
        context: Optional[EvaluationContext] = None,
        use_cache: bool = True,
    ) -> bool:
        """Check if toggle is enabled with caching."""
        cache_key = f"{name}:{hash(str(context.to_dict()) if context else '')}"

        if use_cache and cache_key in self._cache:
            enabled, cached_at = self._cache[cache_key]
            if time.time() - cached_at < self.cache_ttl:
                return enabled

        enabled = await self.manager.is_enabled(name, context)
        self._cache[cache_key] = (enabled, time.time())
        return enabled

    def invalidate_cache(self, name: Optional[str] = None) -> None:
        """Invalidate cache."""
        if name:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{name}:")]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()


def feature_flag(
    name: str,
    manager: ToggleManager,
    default: bool = False,
) -> Callable:
    """Decorator for feature-flagged functions."""
    def decorator(func: Callable) -> Callable:
        import functools

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            context = kwargs.pop("toggle_context", None)
            if await manager.is_enabled(name, context):
                return await func(*args, **kwargs)
            return default

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            context = kwargs.pop("toggle_context", None)
            loop = asyncio.new_event_loop()
            try:
                if loop.run_until_complete(manager.is_enabled(name, context)):
                    return func(*args, **kwargs)
            finally:
                loop.close()
            return default

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
