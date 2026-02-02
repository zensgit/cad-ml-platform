"""Self-Healing Mechanisms.

Provides automatic recovery and healing for common failure scenarios.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class HealingActionType(str, Enum):
    """Types of healing actions."""

    RESTART_CONNECTION = "restart_connection"
    CLEAR_CACHE = "clear_cache"
    CIRCUIT_BREAK = "circuit_break"
    SCALE_UP = "scale_up"
    FAILOVER = "failover"
    RETRY = "retry"
    DEGRADE = "degrade"
    NOTIFY = "notify"


@dataclass
class HealingAction:
    """A healing action to be executed."""

    action_type: HealingActionType
    target: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = more important
    max_attempts: int = 3
    cooldown_seconds: float = 60.0


@dataclass
class HealingResult:
    """Result of a healing action."""

    action: HealingAction
    success: bool
    timestamp: datetime
    message: str
    duration_ms: float
    attempts: int


class HealingStrategy(ABC):
    """Base class for healing strategies."""

    @abstractmethod
    async def diagnose(self, health_result: Any) -> List[HealingAction]:
        """Diagnose issues and return healing actions."""
        pass

    @abstractmethod
    async def heal(self, action: HealingAction) -> HealingResult:
        """Execute a healing action."""
        pass


class ConnectionResetStrategy(HealingStrategy):
    """Strategy for resetting failed connections."""

    def __init__(self, connection_factories: Optional[Dict[str, Callable[[], Any]]] = None):
        self.connection_factories = connection_factories or {}
        self._connections: Dict[str, Any] = {}

    def register_connection(self, name: str, factory: Callable[[], Any]) -> None:
        """Register a connection factory for healing."""
        self.connection_factories[name] = factory

    async def diagnose(self, health_result: Any) -> List[HealingAction]:
        actions = []

        if hasattr(health_result, "dependencies"):
            for name, dep in health_result.dependencies.items():
                if not dep.is_healthy and name in self.connection_factories:
                    actions.append(HealingAction(
                        action_type=HealingActionType.RESTART_CONNECTION,
                        target=name,
                        priority=10,
                    ))

        return actions

    async def heal(self, action: HealingAction) -> HealingResult:
        start_time = time.time()
        attempts = 0

        while attempts < action.max_attempts:
            attempts += 1
            try:
                factory = self.connection_factories.get(action.target)
                if factory:
                    # Close existing connection if any
                    old_conn = self._connections.pop(action.target, None)
                    if old_conn and hasattr(old_conn, "close"):
                        try:
                            if asyncio.iscoroutinefunction(old_conn.close):
                                await old_conn.close()
                            else:
                                old_conn.close()
                        except Exception:
                            pass

                    # Create new connection
                    if asyncio.iscoroutinefunction(factory):
                        new_conn = await factory()
                    else:
                        new_conn = factory()

                    self._connections[action.target] = new_conn

                    return HealingResult(
                        action=action,
                        success=True,
                        timestamp=datetime.utcnow(),
                        message=f"Connection {action.target} reset successfully",
                        duration_ms=(time.time() - start_time) * 1000,
                        attempts=attempts,
                    )

            except Exception as e:
                logger.error(f"Connection reset attempt {attempts} failed: {e}")
                if attempts < action.max_attempts:
                    await asyncio.sleep(1)  # Brief delay between attempts

        return HealingResult(
            action=action,
            success=False,
            timestamp=datetime.utcnow(),
            message=f"Connection reset failed after {attempts} attempts",
            duration_ms=(time.time() - start_time) * 1000,
            attempts=attempts,
        )


class CircuitBreakerHealer(HealingStrategy):
    """Strategy for circuit breaker based healing."""

    def __init__(self):
        self._circuits: Dict[str, Dict[str, Any]] = {}

    def register_circuit(self, name: str, circuit: Any) -> None:
        """Register a circuit breaker for healing."""
        self._circuits[name] = {
            "circuit": circuit,
            "last_reset": None,
        }

    async def diagnose(self, health_result: Any) -> List[HealingAction]:
        actions = []

        for name, info in self._circuits.items():
            circuit = info["circuit"]
            if hasattr(circuit, "state"):
                from src.core.gateway.circuit_breaker import CircuitState
                if circuit.state == CircuitState.OPEN:
                    # Check if we should attempt reset
                    last_reset = info.get("last_reset")
                    if last_reset is None or (time.time() - last_reset) > 60:
                        actions.append(HealingAction(
                            action_type=HealingActionType.CIRCUIT_BREAK,
                            target=name,
                            parameters={"action": "half_open"},
                            priority=5,
                        ))

        return actions

    async def heal(self, action: HealingAction) -> HealingResult:
        start_time = time.time()

        try:
            info = self._circuits.get(action.target)
            if not info:
                return HealingResult(
                    action=action,
                    success=False,
                    timestamp=datetime.utcnow(),
                    message=f"Circuit {action.target} not found",
                    duration_ms=(time.time() - start_time) * 1000,
                    attempts=1,
                )

            circuit = info["circuit"]
            action_type = action.parameters.get("action", "reset")

            if action_type == "half_open":
                # Transition to half-open to test
                from src.core.gateway.circuit_breaker import CircuitState
                if hasattr(circuit, "_transition_to"):
                    circuit._transition_to(CircuitState.HALF_OPEN)
            elif action_type == "reset":
                if hasattr(circuit, "reset"):
                    circuit.reset()

            info["last_reset"] = time.time()

            return HealingResult(
                action=action,
                success=True,
                timestamp=datetime.utcnow(),
                message=f"Circuit {action.target} healing initiated",
                duration_ms=(time.time() - start_time) * 1000,
                attempts=1,
            )

        except Exception as e:
            return HealingResult(
                action=action,
                success=False,
                timestamp=datetime.utcnow(),
                message=f"Circuit healing failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                attempts=1,
            )


class CacheClearStrategy(HealingStrategy):
    """Strategy for clearing caches on issues."""

    def __init__(self):
        self._caches: Dict[str, Any] = {}

    def register_cache(self, name: str, cache: Any) -> None:
        """Register a cache for healing."""
        self._caches[name] = cache

    async def diagnose(self, health_result: Any) -> List[HealingAction]:
        # Could diagnose cache-related issues
        return []

    async def heal(self, action: HealingAction) -> HealingResult:
        start_time = time.time()

        try:
            cache = self._caches.get(action.target)
            if not cache:
                return HealingResult(
                    action=action,
                    success=False,
                    timestamp=datetime.utcnow(),
                    message=f"Cache {action.target} not found",
                    duration_ms=(time.time() - start_time) * 1000,
                    attempts=1,
                )

            if hasattr(cache, "clear"):
                if asyncio.iscoroutinefunction(cache.clear):
                    await cache.clear()
                else:
                    cache.clear()

            return HealingResult(
                action=action,
                success=True,
                timestamp=datetime.utcnow(),
                message=f"Cache {action.target} cleared",
                duration_ms=(time.time() - start_time) * 1000,
                attempts=1,
            )

        except Exception as e:
            return HealingResult(
                action=action,
                success=False,
                timestamp=datetime.utcnow(),
                message=f"Cache clear failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                attempts=1,
            )


class SelfHealer:
    """Central self-healing coordinator."""

    def __init__(self):
        self._strategies: Dict[HealingActionType, HealingStrategy] = {}
        self._action_history: List[HealingResult] = []
        self._cooldowns: Dict[str, float] = {}  # target -> next_allowed_time
        self._running = False
        self._heal_task: Optional[asyncio.Task] = None

    def register_strategy(
        self,
        action_type: HealingActionType,
        strategy: HealingStrategy,
    ) -> None:
        """Register a healing strategy."""
        self._strategies[action_type] = strategy
        logger.info(f"Registered healing strategy for {action_type}")

    async def diagnose(self, health_result: Any) -> List[HealingAction]:
        """Diagnose issues using all strategies."""
        all_actions = []

        for strategy in self._strategies.values():
            try:
                actions = await strategy.diagnose(health_result)
                all_actions.extend(actions)
            except Exception as e:
                logger.error(f"Diagnosis error: {e}")

        # Sort by priority (highest first)
        all_actions.sort(key=lambda a: a.priority, reverse=True)

        return all_actions

    async def heal(self, action: HealingAction) -> HealingResult:
        """Execute a healing action."""
        # Check cooldown
        cooldown_key = f"{action.action_type}:{action.target}"
        if cooldown_key in self._cooldowns:
            if time.time() < self._cooldowns[cooldown_key]:
                return HealingResult(
                    action=action,
                    success=False,
                    timestamp=datetime.utcnow(),
                    message="Action in cooldown",
                    duration_ms=0,
                    attempts=0,
                )

        strategy = self._strategies.get(action.action_type)
        if not strategy:
            return HealingResult(
                action=action,
                success=False,
                timestamp=datetime.utcnow(),
                message=f"No strategy for {action.action_type}",
                duration_ms=0,
                attempts=0,
            )

        result = await strategy.heal(action)

        # Update cooldown
        self._cooldowns[cooldown_key] = time.time() + action.cooldown_seconds

        # Record history
        self._action_history.append(result)
        if len(self._action_history) > 1000:
            self._action_history = self._action_history[-1000:]

        return result

    async def auto_heal(self, health_result: Any) -> List[HealingResult]:
        """Automatically diagnose and heal issues."""
        actions = await self.diagnose(health_result)
        results = []

        for action in actions:
            result = await self.heal(action)
            results.append(result)

            if result.success:
                logger.info(f"Auto-healing succeeded: {action.action_type} on {action.target}")
            else:
                logger.warning(f"Auto-healing failed: {action.action_type} on {action.target}: {result.message}")

        return results

    async def start_auto_healing(
        self,
        health_checker: Any,
        interval_seconds: float = 30.0,
    ) -> None:
        """Start automatic healing loop."""
        if self._running:
            return

        self._running = True

        async def heal_loop() -> None:
            while self._running:
                try:
                    # Get health status
                    health_result = await health_checker.check_all()

                    # Auto-heal if issues detected
                    from src.core.health.checker import HealthStatus
                    if health_result.status != HealthStatus.HEALTHY:
                        await self.auto_heal(health_result)

                except Exception as e:
                    logger.error(f"Auto-healing loop error: {e}")

                await asyncio.sleep(interval_seconds)

        self._heal_task = asyncio.create_task(heal_loop())
        logger.info("Auto-healing started")

    async def stop_auto_healing(self) -> None:
        """Stop automatic healing loop."""
        self._running = False

        if self._heal_task:
            self._heal_task.cancel()
            try:
                await self._heal_task
            except asyncio.CancelledError:
                pass
            self._heal_task = None

        logger.info("Auto-healing stopped")

    def get_history(self, limit: int = 100) -> List[HealingResult]:
        """Get recent healing history."""
        return self._action_history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get healing statistics."""
        total = len(self._action_history)
        successful = sum(1 for r in self._action_history if r.success)

        by_type: Dict[str, Dict[str, int]] = {}
        for result in self._action_history:
            action_type = result.action.action_type.value
            if action_type not in by_type:
                by_type[action_type] = {"total": 0, "success": 0}
            by_type[action_type]["total"] += 1
            if result.success:
                by_type[action_type]["success"] += 1

        return {
            "total_actions": total,
            "successful_actions": successful,
            "success_rate": successful / total if total > 0 else 0,
            "by_type": by_type,
            "active_cooldowns": len(self._cooldowns),
        }


# Global self-healer
_self_healer: Optional[SelfHealer] = None


def get_self_healer() -> SelfHealer:
    """Get global self-healer."""
    global _self_healer
    if _self_healer is None:
        _self_healer = SelfHealer()
    return _self_healer


def setup_default_healing(healer: Optional[SelfHealer] = None) -> SelfHealer:
    """Setup default healing strategies."""
    sh = healer or get_self_healer()

    sh.register_strategy(
        HealingActionType.RESTART_CONNECTION,
        ConnectionResetStrategy(),
    )
    sh.register_strategy(
        HealingActionType.CIRCUIT_BREAK,
        CircuitBreakerHealer(),
    )
    sh.register_strategy(
        HealingActionType.CLEAR_CACHE,
        CacheClearStrategy(),
    )

    return sh
