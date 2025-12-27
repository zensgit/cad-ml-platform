"""Intelligent Automation & Self-Optimization Module for Vision System.

This module provides intelligent automation capabilities including:
- Automated decision engine with rule-based and ML-based decisions
- Self-tuning parameter optimization
- Intelligent task scheduler with priority and resource awareness
- Adaptive load management
- Performance prediction and forecasting
- Automatic remediation and recovery
- Learning from operational patterns

Phase 23: Intelligent Automation & Self-Optimization
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import statistics
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from .base import VisionDescription, VisionProvider

# ========================
# Enums
# ========================


class DecisionType(str, Enum):
    """Types of automated decisions."""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    ROUTE_TRAFFIC = "route_traffic"
    FAILOVER = "failover"
    OPTIMIZE = "optimize"
    ALERT = "alert"
    REMEDIATE = "remediate"
    DEFER = "defer"


class DecisionConfidence(str, Enum):
    """Confidence levels for decisions."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


class TuningStrategy(str, Enum):
    """Parameter tuning strategies."""

    GRADIENT_DESCENT = "gradient_descent"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    ADAPTIVE = "adaptive"


class TuningStatus(str, Enum):
    """Status of tuning operations."""

    IDLE = "idle"
    EXPLORING = "exploring"
    EXPLOITING = "exploiting"
    CONVERGED = "converged"
    FAILED = "failed"


class SchedulerPriority(str, Enum):
    """Task scheduler priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class TaskState(str, Enum):
    """Scheduled task states."""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEFERRED = "deferred"


class LoadLevel(str, Enum):
    """System load levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    IDLE = "idle"


class RemediationAction(str, Enum):
    """Types of remediation actions."""

    RESTART = "restart"
    SCALE = "scale"
    FAILOVER = "failover"
    THROTTLE = "throttle"
    CLEAR_CACHE = "clear_cache"
    ROLLBACK = "rollback"
    NOTIFY = "notify"
    CUSTOM = "custom"


class PredictionType(str, Enum):
    """Types of predictions."""

    LOAD = "load"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    THROUGHPUT = "throughput"
    COST = "cost"


class LearningMode(str, Enum):
    """Learning modes for the system."""

    ONLINE = "online"
    BATCH = "batch"
    REINFORCEMENT = "reinforcement"
    SUPERVISED = "supervised"


# ========================
# Dataclasses
# ========================


@dataclass
class Decision:
    """Represents an automated decision."""

    decision_id: str
    decision_type: DecisionType
    confidence: DecisionConfidence
    rationale: str
    parameters: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    executed: bool = False
    outcome: Optional[str] = None
    feedback_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type.value,
            "confidence": self.confidence.value,
            "rationale": self.rationale,
            "parameters": self.parameters,
            "timestamp": self.timestamp.isoformat(),
            "executed": self.executed,
            "outcome": self.outcome,
            "feedback_score": self.feedback_score,
        }


@dataclass
class DecisionRule:
    """Rule for automated decision making."""

    rule_id: str
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    decision_type: DecisionType
    parameters_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    priority: int = 0
    enabled: bool = True
    description: str = ""


@dataclass
class TuningParameter:
    """A parameter that can be tuned."""

    name: str
    current_value: float
    min_value: float
    max_value: float
    step_size: float = 0.1
    best_value: Optional[float] = None
    best_score: Optional[float] = None
    history: List[Tuple[float, float]] = field(default_factory=list)

    def get_next_value(self, strategy: TuningStrategy) -> float:
        """Get next value to try based on strategy."""
        if strategy == TuningStrategy.RANDOM_SEARCH:
            return random.uniform(self.min_value, self.max_value)
        elif strategy == TuningStrategy.GRID_SEARCH:
            # Move to next grid point
            next_val = self.current_value + self.step_size
            if next_val > self.max_value:
                next_val = self.min_value
            return next_val
        elif strategy == TuningStrategy.GRADIENT_DESCENT:
            # Simple gradient approximation
            if len(self.history) >= 2:
                v1, s1 = self.history[-2]
                v2, s2 = self.history[-1]
                if v2 != v1:
                    gradient = (s2 - s1) / (v2 - v1)
                    next_val = self.current_value - 0.1 * gradient
                    return max(self.min_value, min(self.max_value, next_val))
            return self.current_value + random.uniform(-self.step_size, self.step_size)
        else:
            # Adaptive: explore if uncertain, exploit if confident
            if self.best_value is not None and random.random() > 0.3:
                # Exploit around best
                delta = random.gauss(0, self.step_size)
                return max(self.min_value, min(self.max_value, self.best_value + delta))
            else:
                # Explore
                return random.uniform(self.min_value, self.max_value)


@dataclass
class TuningSession:
    """A parameter tuning session."""

    session_id: str
    parameters: Dict[str, TuningParameter]
    strategy: TuningStrategy
    status: TuningStatus
    objective_fn: Optional[Callable[[Dict[str, float]], float]] = None
    iterations: int = 0
    max_iterations: int = 100
    convergence_threshold: float = 0.001
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class ScheduledTask:
    """A task scheduled for execution."""

    task_id: str
    name: str
    priority: SchedulerPriority
    state: TaskState
    execute_fn: Callable[[], Any]
    scheduled_time: datetime
    deadline: Optional[datetime] = None
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    retries: int = 0
    max_retries: int = 3
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class ResourcePool:
    """Available resources for task scheduling."""

    cpu: float = 100.0  # percentage available
    memory: float = 100.0  # percentage available
    io_bandwidth: float = 100.0  # percentage available
    network: float = 100.0  # percentage available
    custom: Dict[str, float] = field(default_factory=dict)

    def can_allocate(self, requirements: Dict[str, float]) -> bool:
        """Check if requirements can be satisfied."""
        for resource, required in requirements.items():
            if resource == "cpu" and self.cpu < required:
                return False
            elif resource == "memory" and self.memory < required:
                return False
            elif resource == "io_bandwidth" and self.io_bandwidth < required:
                return False
            elif resource == "network" and self.network < required:
                return False
            elif resource in self.custom and self.custom[resource] < required:
                return False
        return True

    def allocate(self, requirements: Dict[str, float]) -> None:
        """Allocate resources."""
        for resource, required in requirements.items():
            if resource == "cpu":
                self.cpu -= required
            elif resource == "memory":
                self.memory -= required
            elif resource == "io_bandwidth":
                self.io_bandwidth -= required
            elif resource == "network":
                self.network -= required
            elif resource in self.custom:
                self.custom[resource] -= required

    def release(self, requirements: Dict[str, float]) -> None:
        """Release resources."""
        for resource, amount in requirements.items():
            if resource == "cpu":
                self.cpu = min(100.0, self.cpu + amount)
            elif resource == "memory":
                self.memory = min(100.0, self.memory + amount)
            elif resource == "io_bandwidth":
                self.io_bandwidth = min(100.0, self.io_bandwidth + amount)
            elif resource == "network":
                self.network = min(100.0, self.network + amount)
            elif resource in self.custom:
                self.custom[resource] = min(100.0, self.custom[resource] + amount)


@dataclass
class LoadMetrics:
    """System load metrics."""

    cpu_usage: float
    memory_usage: float
    request_rate: float
    error_rate: float
    latency_p50: float
    latency_p99: float
    queue_depth: int
    timestamp: datetime = field(default_factory=datetime.now)

    def get_load_level(self) -> LoadLevel:
        """Determine overall load level."""
        # Simple heuristic based on CPU and memory
        avg_usage = (self.cpu_usage + self.memory_usage) / 2
        if avg_usage > 90 or self.error_rate > 0.1:
            return LoadLevel.CRITICAL
        elif avg_usage > 75 or self.error_rate > 0.05:
            return LoadLevel.HIGH
        elif avg_usage > 50:
            return LoadLevel.MODERATE
        elif avg_usage > 20:
            return LoadLevel.LOW
        else:
            return LoadLevel.IDLE


@dataclass
class Remediation:
    """A remediation action record."""

    remediation_id: str
    action: RemediationAction
    target: str
    reason: str
    parameters: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = False
    result: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class Prediction:
    """A prediction result."""

    prediction_id: str
    prediction_type: PredictionType
    target_time: datetime
    predicted_value: float
    confidence_interval: Tuple[float, float]
    confidence: float
    model_version: str
    timestamp: datetime = field(default_factory=datetime.now)
    actual_value: Optional[float] = None


@dataclass
class LearningPattern:
    """A learned operational pattern."""

    pattern_id: str
    name: str
    conditions: Dict[str, Any]
    optimal_action: str
    success_rate: float
    sample_count: int
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class AutomationConfig:
    """Configuration for intelligent automation."""

    decision_cooldown_seconds: int = 60
    tuning_enabled: bool = True
    prediction_enabled: bool = True
    auto_remediation_enabled: bool = True
    learning_enabled: bool = True
    max_concurrent_tasks: int = 10
    load_threshold_critical: float = 0.9
    load_threshold_high: float = 0.75
    prediction_horizon_minutes: int = 30


# ========================
# Core Classes
# ========================


class DecisionEngine:
    """Automated decision engine with rule-based and learned decisions."""

    def __init__(self, config: Optional[AutomationConfig] = None) -> None:
        """Initialize decision engine."""
        self._config = config or AutomationConfig()
        self._rules: Dict[str, DecisionRule] = {}
        self._decisions: List[Decision] = []
        self._last_decision_time: Dict[str, datetime] = {}
        self._lock = threading.RLock()

    def add_rule(self, rule: DecisionRule) -> None:
        """Add a decision rule."""
        with self._lock:
            self._rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a decision rule."""
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                return True
            return False

    def get_rule(self, rule_id: str) -> Optional[DecisionRule]:
        """Get a rule by ID."""
        with self._lock:
            return self._rules.get(rule_id)

    def list_rules(self) -> List[DecisionRule]:
        """List all rules."""
        with self._lock:
            return list(self._rules.values())

    def evaluate(self, context: Dict[str, Any]) -> Optional[Decision]:
        """Evaluate context against rules and make a decision."""
        with self._lock:
            # Sort rules by priority (higher first)
            sorted_rules = sorted(
                [r for r in self._rules.values() if r.enabled],
                key=lambda r: r.priority,
                reverse=True,
            )

            for rule in sorted_rules:
                try:
                    if rule.condition(context):
                        # Check cooldown
                        last_time = self._last_decision_time.get(rule.rule_id)
                        if last_time:
                            elapsed = (datetime.now() - last_time).total_seconds()
                            if elapsed < self._config.decision_cooldown_seconds:
                                continue

                        # Create decision
                        decision = Decision(
                            decision_id=str(uuid.uuid4()),
                            decision_type=rule.decision_type,
                            confidence=DecisionConfidence.HIGH,
                            rationale=f"Rule '{rule.name}' matched: {rule.description}",
                            parameters=rule.parameters_fn(context),
                        )

                        self._decisions.append(decision)
                        self._last_decision_time[rule.rule_id] = datetime.now()
                        return decision
                except Exception:
                    continue

            return None

    def record_feedback(self, decision_id: str, outcome: str, score: float) -> Optional[Decision]:
        """Record feedback for a decision."""
        with self._lock:
            for decision in self._decisions:
                if decision.decision_id == decision_id:
                    decision.outcome = outcome
                    decision.feedback_score = score
                    return decision
            return None

    def get_decision_history(self, limit: int = 100) -> List[Decision]:
        """Get recent decisions."""
        with self._lock:
            return self._decisions[-limit:]

    def get_decision_stats(self) -> Dict[str, Any]:
        """Get decision statistics."""
        with self._lock:
            if not self._decisions:
                return {
                    "total_decisions": 0,
                    "by_type": {},
                    "avg_feedback_score": None,
                }

            by_type: Dict[str, int] = defaultdict(int)
            scores = []
            for d in self._decisions:
                by_type[d.decision_type.value] += 1
                if d.feedback_score is not None:
                    scores.append(d.feedback_score)

            return {
                "total_decisions": len(self._decisions),
                "by_type": dict(by_type),
                "avg_feedback_score": statistics.mean(scores) if scores else None,
            }


class SelfTuner:
    """Self-tuning parameter optimizer."""

    def __init__(self, config: Optional[AutomationConfig] = None) -> None:
        """Initialize self-tuner."""
        self._config = config or AutomationConfig()
        self._sessions: Dict[str, TuningSession] = {}
        self._lock = threading.RLock()

    def create_session(
        self,
        parameters: Dict[str, Tuple[float, float, float]],  # name: (min, max, initial)
        strategy: TuningStrategy = TuningStrategy.ADAPTIVE,
        objective_fn: Optional[Callable[[Dict[str, float]], float]] = None,
        max_iterations: int = 100,
    ) -> TuningSession:
        """Create a new tuning session."""
        with self._lock:
            session_id = str(uuid.uuid4())
            tuning_params = {}
            for name, (min_val, max_val, initial) in parameters.items():
                tuning_params[name] = TuningParameter(
                    name=name,
                    current_value=initial,
                    min_value=min_val,
                    max_value=max_val,
                )

            session = TuningSession(
                session_id=session_id,
                parameters=tuning_params,
                strategy=strategy,
                status=TuningStatus.IDLE,
                objective_fn=objective_fn,
                max_iterations=max_iterations,
            )
            self._sessions[session_id] = session
            return session

    def get_session(self, session_id: str) -> Optional[TuningSession]:
        """Get a tuning session."""
        with self._lock:
            return self._sessions.get(session_id)

    def start_session(self, session_id: str) -> bool:
        """Start a tuning session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.status == TuningStatus.IDLE:
                session.status = TuningStatus.EXPLORING
                return True
            return False

    def step(self, session_id: str) -> Optional[Dict[str, float]]:
        """Perform one tuning step, returns next parameter values to try."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or session.status not in (
                TuningStatus.EXPLORING,
                TuningStatus.EXPLOITING,
            ):
                return None

            if session.iterations >= session.max_iterations:
                session.status = TuningStatus.CONVERGED
                session.completed_at = datetime.now()
                return None

            # Generate next values
            next_values = {}
            for name, param in session.parameters.items():
                next_values[name] = param.get_next_value(session.strategy)
                param.current_value = next_values[name]

            session.iterations += 1
            return next_values

    def record_result(self, session_id: str, values: Dict[str, float], score: float) -> bool:
        """Record the result of a tuning iteration."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            for name, value in values.items():
                if name in session.parameters:
                    param = session.parameters[name]
                    param.history.append((value, score))
                    if param.best_score is None or score > param.best_score:
                        param.best_score = score
                        param.best_value = value

            # Check convergence
            if len(session.parameters) > 0:
                all_converged = True
                for param in session.parameters.values():
                    if len(param.history) < 10:
                        all_converged = False
                        break
                    recent_scores = [s for _, s in param.history[-10:]]
                    if max(recent_scores) - min(recent_scores) > session.convergence_threshold:
                        all_converged = False
                        break

                if all_converged:
                    session.status = TuningStatus.CONVERGED
                    session.completed_at = datetime.now()
                elif session.iterations > session.max_iterations // 2:
                    session.status = TuningStatus.EXPLOITING

            return True

    def get_best_parameters(self, session_id: str) -> Optional[Dict[str, float]]:
        """Get the best parameters found so far."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            best = {}
            for name, param in session.parameters.items():
                if param.best_value is not None:
                    best[name] = param.best_value
                else:
                    best[name] = param.current_value
            return best

    def list_sessions(self) -> List[TuningSession]:
        """List all tuning sessions."""
        with self._lock:
            return list(self._sessions.values())


class IntelligentScheduler:
    """Intelligent task scheduler with priority and resource awareness."""

    def __init__(self, config: Optional[AutomationConfig] = None) -> None:
        """Initialize scheduler."""
        self._config = config or AutomationConfig()
        self._tasks: Dict[str, ScheduledTask] = {}
        self._resource_pool = ResourcePool()
        self._running_tasks: Set[str] = set()
        self._lock = threading.RLock()

    def schedule_task(
        self,
        name: str,
        execute_fn: Callable[[], Any],
        priority: SchedulerPriority = SchedulerPriority.NORMAL,
        scheduled_time: Optional[datetime] = None,
        deadline: Optional[datetime] = None,
        resource_requirements: Optional[Dict[str, float]] = None,
        dependencies: Optional[List[str]] = None,
    ) -> ScheduledTask:
        """Schedule a new task."""
        with self._lock:
            task_id = str(uuid.uuid4())
            task = ScheduledTask(
                task_id=task_id,
                name=name,
                priority=priority,
                state=TaskState.PENDING,
                execute_fn=execute_fn,
                scheduled_time=scheduled_time or datetime.now(),
                deadline=deadline,
                resource_requirements=resource_requirements or {},
                dependencies=dependencies or [],
            )
            self._tasks[task_id] = task
            return task

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.state in (TaskState.PENDING, TaskState.SCHEDULED):
                task.state = TaskState.CANCELLED
                return True
            return False

    def get_next_task(self) -> Optional[ScheduledTask]:
        """Get the next task to execute based on priority and dependencies."""
        with self._lock:
            now = datetime.now()

            # Find eligible tasks
            eligible = []
            for task in self._tasks.values():
                if task.state != TaskState.PENDING:
                    continue
                if task.scheduled_time > now:
                    continue
                if len(self._running_tasks) >= self._config.max_concurrent_tasks:
                    continue
                if not self._resource_pool.can_allocate(task.resource_requirements):
                    continue
                # Check dependencies
                deps_met = True
                for dep_id in task.dependencies:
                    dep_task = self._tasks.get(dep_id)
                    if not dep_task or dep_task.state != TaskState.COMPLETED:
                        deps_met = False
                        break
                if not deps_met:
                    continue
                eligible.append(task)

            if not eligible:
                return None

            # Sort by priority and deadline
            def sort_key(t: ScheduledTask) -> Tuple[int, float]:
                priority_order = {
                    SchedulerPriority.CRITICAL: 0,
                    SchedulerPriority.HIGH: 1,
                    SchedulerPriority.NORMAL: 2,
                    SchedulerPriority.LOW: 3,
                    SchedulerPriority.BACKGROUND: 4,
                }
                deadline_urgency = 0.0 if t.deadline is None else (t.deadline - now).total_seconds()
                return (priority_order[t.priority], deadline_urgency)

            eligible.sort(key=sort_key)
            return eligible[0]

    def start_task(self, task_id: str) -> bool:
        """Mark a task as started."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.state == TaskState.PENDING:
                task.state = TaskState.RUNNING
                task.started_at = datetime.now()
                self._running_tasks.add(task_id)
                self._resource_pool.allocate(task.resource_requirements)
                return True
            return False

    def complete_task(self, task_id: str, result: Any = None, error: Optional[str] = None) -> bool:
        """Mark a task as completed."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.state == TaskState.RUNNING:
                task.state = TaskState.COMPLETED if error is None else TaskState.FAILED
                task.completed_at = datetime.now()
                task.result = result
                task.error = error
                self._running_tasks.discard(task_id)
                self._resource_pool.release(task.resource_requirements)
                return True
            return False

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get scheduler queue statistics."""
        with self._lock:
            by_state: Dict[str, int] = defaultdict(int)
            by_priority: Dict[str, int] = defaultdict(int)
            for task in self._tasks.values():
                by_state[task.state.value] += 1
                by_priority[task.priority.value] += 1

            return {
                "total_tasks": len(self._tasks),
                "running_tasks": len(self._running_tasks),
                "by_state": dict(by_state),
                "by_priority": dict(by_priority),
                "resource_pool": {
                    "cpu": self._resource_pool.cpu,
                    "memory": self._resource_pool.memory,
                },
            }

    def list_tasks(
        self, state: Optional[TaskState] = None, limit: int = 100
    ) -> List[ScheduledTask]:
        """List tasks, optionally filtered by state."""
        with self._lock:
            tasks = list(self._tasks.values())
            if state:
                tasks = [t for t in tasks if t.state == state]
            return tasks[:limit]


class LoadManager:
    """Adaptive load management system."""

    def __init__(self, config: Optional[AutomationConfig] = None) -> None:
        """Initialize load manager."""
        self._config = config or AutomationConfig()
        self._metrics_history: List[LoadMetrics] = []
        self._load_handlers: Dict[LoadLevel, List[Callable[[LoadMetrics], None]]] = defaultdict(
            list
        )
        self._lock = threading.RLock()
        self._max_history = 1000

    def record_metrics(self, metrics: LoadMetrics) -> LoadLevel:
        """Record load metrics and return current load level."""
        with self._lock:
            self._metrics_history.append(metrics)
            if len(self._metrics_history) > self._max_history:
                self._metrics_history = self._metrics_history[-self._max_history :]

            load_level = metrics.get_load_level()

            # Trigger handlers
            for handler in self._load_handlers.get(load_level, []):
                try:
                    handler(metrics)
                except Exception:
                    pass

            return load_level

    def register_handler(self, level: LoadLevel, handler: Callable[[LoadMetrics], None]) -> None:
        """Register a handler for a specific load level."""
        with self._lock:
            self._load_handlers[level].append(handler)

    def get_current_load(self) -> Optional[LoadLevel]:
        """Get current load level."""
        with self._lock:
            if self._metrics_history:
                return self._metrics_history[-1].get_load_level()
            return None

    def get_average_metrics(self, window_seconds: int = 300) -> Optional[LoadMetrics]:
        """Get average metrics over a time window."""
        with self._lock:
            if not self._metrics_history:
                return None

            cutoff = datetime.now() - timedelta(seconds=window_seconds)
            recent = [m for m in self._metrics_history if m.timestamp >= cutoff]

            if not recent:
                return None

            return LoadMetrics(
                cpu_usage=statistics.mean([m.cpu_usage for m in recent]),
                memory_usage=statistics.mean([m.memory_usage for m in recent]),
                request_rate=statistics.mean([m.request_rate for m in recent]),
                error_rate=statistics.mean([m.error_rate for m in recent]),
                latency_p50=statistics.mean([m.latency_p50 for m in recent]),
                latency_p99=statistics.mean([m.latency_p99 for m in recent]),
                queue_depth=int(statistics.mean([m.queue_depth for m in recent])),
            )

    def get_metrics_history(self, limit: int = 100) -> List[LoadMetrics]:
        """Get metrics history."""
        with self._lock:
            return self._metrics_history[-limit:]

    def should_shed_load(self) -> bool:
        """Determine if load shedding should be activated."""
        with self._lock:
            current = self.get_current_load()
            return current in (LoadLevel.CRITICAL, LoadLevel.HIGH)

    def get_throttle_percentage(self) -> float:
        """Get recommended throttle percentage (0-100)."""
        with self._lock:
            if not self._metrics_history:
                return 0.0

            metrics = self._metrics_history[-1]
            avg_usage = (metrics.cpu_usage + metrics.memory_usage) / 2

            if avg_usage > 95:
                return 50.0
            elif avg_usage > 90:
                return 30.0
            elif avg_usage > 85:
                return 20.0
            elif avg_usage > 80:
                return 10.0
            else:
                return 0.0


class PerformancePredictor:
    """Performance prediction and forecasting system."""

    def __init__(self, config: Optional[AutomationConfig] = None) -> None:
        """Initialize predictor."""
        self._config = config or AutomationConfig()
        self._historical_data: Dict[PredictionType, List[Tuple[datetime, float]]] = defaultdict(
            list
        )
        self._predictions: List[Prediction] = []
        self._lock = threading.RLock()
        self._max_history = 10000

    def record_observation(self, prediction_type: PredictionType, value: float) -> None:
        """Record an observation for a metric type."""
        with self._lock:
            history = self._historical_data[prediction_type]
            history.append((datetime.now(), value))
            if len(history) > self._max_history:
                self._historical_data[prediction_type] = history[-self._max_history :]

    def predict(
        self,
        prediction_type: PredictionType,
        horizon_minutes: Optional[int] = None,
    ) -> Optional[Prediction]:
        """Make a prediction for a metric type."""
        with self._lock:
            horizon = horizon_minutes or self._config.prediction_horizon_minutes
            history = self._historical_data.get(prediction_type, [])

            if len(history) < 10:
                return None

            # Simple linear regression for prediction
            recent = history[-100:]
            values = [v for _, v in recent]
            times = [(t - recent[0][0]).total_seconds() for t, _ in recent]

            # Calculate trend
            n = len(values)
            sum_x = sum(times)
            sum_y = sum(values)
            sum_xy = sum(t * v for t, v in zip(times, values))
            sum_xx = sum(t * t for t in times)

            denom = n * sum_xx - sum_x * sum_x
            if abs(denom) < 1e-10:
                slope = 0
                intercept = statistics.mean(values)
            else:
                slope = (n * sum_xy - sum_x * sum_y) / denom
                intercept = (sum_y - slope * sum_x) / n

            # Predict future value
            current_time = (datetime.now() - recent[0][0]).total_seconds()
            future_time = current_time + horizon * 60
            predicted_value = slope * future_time + intercept

            # Calculate confidence interval based on historical variance
            std_dev = statistics.stdev(values) if len(values) > 1 else 0
            confidence_interval = (
                predicted_value - 2 * std_dev,
                predicted_value + 2 * std_dev,
            )

            # Confidence based on data quality
            confidence = min(0.95, 0.5 + len(history) / 1000)

            prediction = Prediction(
                prediction_id=str(uuid.uuid4()),
                prediction_type=prediction_type,
                target_time=datetime.now() + timedelta(minutes=horizon),
                predicted_value=predicted_value,
                confidence_interval=confidence_interval,
                confidence=confidence,
                model_version="linear_v1",
            )

            self._predictions.append(prediction)
            return prediction

    def validate_prediction(self, prediction_id: str, actual_value: float) -> Optional[Prediction]:
        """Validate a prediction with the actual value."""
        with self._lock:
            for pred in self._predictions:
                if pred.prediction_id == prediction_id:
                    pred.actual_value = actual_value
                    return pred
            return None

    def get_prediction_accuracy(
        self, prediction_type: Optional[PredictionType] = None
    ) -> Dict[str, float]:
        """Get prediction accuracy metrics."""
        with self._lock:
            validated = [
                p
                for p in self._predictions
                if p.actual_value is not None
                and (prediction_type is None or p.prediction_type == prediction_type)
            ]

            if not validated:
                return {"mape": None, "rmse": None, "count": 0}

            errors = []
            for p in validated:
                if p.actual_value != 0:
                    errors.append(abs(p.predicted_value - p.actual_value) / abs(p.actual_value))

            mape = statistics.mean(errors) * 100 if errors else None

            squared_errors = [(p.predicted_value - p.actual_value) ** 2 for p in validated]
            rmse = math.sqrt(statistics.mean(squared_errors)) if squared_errors else None

            return {"mape": mape, "rmse": rmse, "count": len(validated)}

    def get_predictions(self, limit: int = 100) -> List[Prediction]:
        """Get recent predictions."""
        with self._lock:
            return self._predictions[-limit:]


class AutoRemediation:
    """Automatic remediation and recovery system."""

    def __init__(self, config: Optional[AutomationConfig] = None) -> None:
        """Initialize auto-remediation."""
        self._config = config or AutomationConfig()
        self._remediation_handlers: Dict[
            RemediationAction, Callable[[str, Dict[str, Any]], bool]
        ] = {}
        self._remediations: List[Remediation] = []
        self._lock = threading.RLock()

    def register_handler(
        self,
        action: RemediationAction,
        handler: Callable[[str, Dict[str, Any]], bool],
    ) -> None:
        """Register a remediation handler."""
        with self._lock:
            self._remediation_handlers[action] = handler

    def execute_remediation(
        self,
        action: RemediationAction,
        target: str,
        reason: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Remediation:
        """Execute a remediation action."""
        with self._lock:
            if not self._config.auto_remediation_enabled:
                remediation = Remediation(
                    remediation_id=str(uuid.uuid4()),
                    action=action,
                    target=target,
                    reason=reason,
                    parameters=parameters or {},
                    success=False,
                    result="Auto-remediation disabled",
                )
                self._remediations.append(remediation)
                return remediation

            start_time = time.time()
            success = False
            result = "No handler registered"

            handler = self._remediation_handlers.get(action)
            if handler:
                try:
                    success = handler(target, parameters or {})
                    result = "Success" if success else "Handler returned false"
                except Exception as e:
                    result = f"Error: {str(e)}"

            duration_ms = (time.time() - start_time) * 1000

            remediation = Remediation(
                remediation_id=str(uuid.uuid4()),
                action=action,
                target=target,
                reason=reason,
                parameters=parameters or {},
                success=success,
                result=result,
                duration_ms=duration_ms,
            )
            self._remediations.append(remediation)
            return remediation

    def get_remediation_history(
        self, limit: int = 100, success_only: bool = False
    ) -> List[Remediation]:
        """Get remediation history."""
        with self._lock:
            remediations = self._remediations
            if success_only:
                remediations = [r for r in remediations if r.success]
            return remediations[-limit:]

    def get_remediation_stats(self) -> Dict[str, Any]:
        """Get remediation statistics."""
        with self._lock:
            if not self._remediations:
                return {"total": 0, "success_rate": None, "by_action": {}}

            by_action: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})
            for r in self._remediations:
                by_action[r.action.value]["total"] += 1
                if r.success:
                    by_action[r.action.value]["success"] += 1

            total = len(self._remediations)
            success = sum(1 for r in self._remediations if r.success)

            return {
                "total": total,
                "success_rate": success / total if total > 0 else None,
                "by_action": dict(by_action),
            }


class PatternLearner:
    """Learning system for operational patterns."""

    def __init__(self, config: Optional[AutomationConfig] = None) -> None:
        """Initialize pattern learner."""
        self._config = config or AutomationConfig()
        self._patterns: Dict[str, LearningPattern] = {}
        self._observations: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._max_observations = 10000

    def record_observation(
        self,
        conditions: Dict[str, Any],
        action: str,
        success: bool,
    ) -> None:
        """Record an observation of conditions, action, and outcome."""
        with self._lock:
            if not self._config.learning_enabled:
                return

            observation = {
                "conditions": conditions,
                "action": action,
                "success": success,
                "timestamp": datetime.now(),
            }
            self._observations.append(observation)

            if len(self._observations) > self._max_observations:
                self._observations = self._observations[-self._max_observations :]

            # Update or create pattern
            pattern_key = self._generate_pattern_key(conditions)
            if pattern_key in self._patterns:
                pattern = self._patterns[pattern_key]
                # Update success rate with exponential moving average
                pattern.sample_count += 1
                alpha = 0.1
                current_success = 1.0 if success else 0.0
                pattern.success_rate = alpha * current_success + (1 - alpha) * pattern.success_rate
                if success and pattern.success_rate > 0.5:
                    pattern.optimal_action = action
                pattern.last_updated = datetime.now()
            else:
                self._patterns[pattern_key] = LearningPattern(
                    pattern_id=pattern_key,
                    name=f"Pattern_{len(self._patterns) + 1}",
                    conditions=conditions,
                    optimal_action=action,
                    success_rate=1.0 if success else 0.0,
                    sample_count=1,
                )

    def _generate_pattern_key(self, conditions: Dict[str, Any]) -> str:
        """Generate a unique key for conditions."""
        sorted_items = sorted(conditions.items())
        return hashlib.md5(json.dumps(sorted_items).encode()).hexdigest()[:16]

    def get_recommended_action(self, conditions: Dict[str, Any]) -> Optional[Tuple[str, float]]:
        """Get recommended action for given conditions."""
        with self._lock:
            pattern_key = self._generate_pattern_key(conditions)
            pattern = self._patterns.get(pattern_key)
            if pattern and pattern.sample_count >= 5:
                return (pattern.optimal_action, pattern.success_rate)
            return None

    def get_patterns(self, min_samples: int = 5) -> List[LearningPattern]:
        """Get learned patterns with minimum sample count."""
        with self._lock:
            return [p for p in self._patterns.values() if p.sample_count >= min_samples]

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        with self._lock:
            patterns = list(self._patterns.values())
            if not patterns:
                return {
                    "total_patterns": 0,
                    "total_observations": len(self._observations),
                    "avg_success_rate": None,
                }

            return {
                "total_patterns": len(patterns),
                "total_observations": len(self._observations),
                "avg_success_rate": statistics.mean([p.success_rate for p in patterns]),
                "patterns_with_high_confidence": sum(1 for p in patterns if p.sample_count >= 10),
            }


class IntelligentAutomationHub:
    """Central hub for intelligent automation and self-optimization."""

    def __init__(self, config: Optional[AutomationConfig] = None) -> None:
        """Initialize automation hub."""
        self._config = config or AutomationConfig()
        self._decision_engine = DecisionEngine(self._config)
        self._self_tuner = SelfTuner(self._config)
        self._scheduler = IntelligentScheduler(self._config)
        self._load_manager = LoadManager(self._config)
        self._predictor = PerformancePredictor(self._config)
        self._remediation = AutoRemediation(self._config)
        self._learner = PatternLearner(self._config)
        self._lock = threading.RLock()

    @property
    def decision_engine(self) -> DecisionEngine:
        """Get decision engine."""
        return self._decision_engine

    @property
    def self_tuner(self) -> SelfTuner:
        """Get self-tuner."""
        return self._self_tuner

    @property
    def scheduler(self) -> IntelligentScheduler:
        """Get intelligent scheduler."""
        return self._scheduler

    @property
    def load_manager(self) -> LoadManager:
        """Get load manager."""
        return self._load_manager

    @property
    def predictor(self) -> PerformancePredictor:
        """Get performance predictor."""
        return self._predictor

    @property
    def remediation(self) -> AutoRemediation:
        """Get auto-remediation."""
        return self._remediation

    @property
    def learner(self) -> PatternLearner:
        """Get pattern learner."""
        return self._learner

    def process_metrics(self, metrics: LoadMetrics) -> Dict[str, Any]:
        """Process metrics through the automation pipeline."""
        with self._lock:
            results: Dict[str, Any] = {}

            # Record load metrics
            load_level = self._load_manager.record_metrics(metrics)
            results["load_level"] = load_level.value

            # Record observations for prediction
            self._predictor.record_observation(PredictionType.LOAD, metrics.cpu_usage)
            self._predictor.record_observation(PredictionType.LATENCY, metrics.latency_p99)
            self._predictor.record_observation(PredictionType.ERROR_RATE, metrics.error_rate)

            # Evaluate decision rules
            context = {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "error_rate": metrics.error_rate,
                "latency_p99": metrics.latency_p99,
                "load_level": load_level.value,
            }
            decision = self._decision_engine.evaluate(context)
            if decision:
                results["decision"] = decision.to_dict()

            # Get predictions
            if self._config.prediction_enabled:
                load_pred = self._predictor.predict(PredictionType.LOAD)
                if load_pred:
                    results["load_prediction"] = {
                        "value": load_pred.predicted_value,
                        "confidence": load_pred.confidence,
                    }

            # Check if remediation needed
            if load_level == LoadLevel.CRITICAL and self._config.auto_remediation_enabled:
                remediation = self._remediation.execute_remediation(
                    action=RemediationAction.THROTTLE,
                    target="system",
                    reason="Critical load level detected",
                    parameters={"throttle_percentage": 30},
                )
                results["remediation"] = {
                    "action": remediation.action.value,
                    "success": remediation.success,
                }

            return results

    def get_automation_summary(self) -> Dict[str, Any]:
        """Get a summary of automation state."""
        with self._lock:
            return {
                "decision_stats": self._decision_engine.get_decision_stats(),
                "scheduler_stats": self._scheduler.get_queue_stats(),
                "current_load": (
                    self._load_manager.get_current_load().value
                    if self._load_manager.get_current_load()
                    else None
                ),
                "prediction_accuracy": self._predictor.get_prediction_accuracy(),
                "remediation_stats": self._remediation.get_remediation_stats(),
                "learning_stats": self._learner.get_learning_stats(),
                "tuning_sessions": len(self._self_tuner.list_sessions()),
            }


class AutomatedVisionProvider(VisionProvider):
    """Vision provider with intelligent automation capabilities."""

    def __init__(
        self,
        base_provider: VisionProvider,
        hub: Optional[IntelligentAutomationHub] = None,
    ) -> None:
        """Initialize automated provider."""
        self._base_provider = base_provider
        self._hub = hub or IntelligentAutomationHub()
        self._request_count = 0
        self._error_count = 0
        self._total_latency = 0.0
        self._lock = threading.RLock()

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"automated_{self._base_provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with automation tracking."""
        start_time = time.time()
        error_occurred = False

        try:
            # Check load shedding
            if self._hub.load_manager.should_shed_load():
                throttle = self._hub.load_manager.get_throttle_percentage()
                if random.random() * 100 < throttle:
                    # Record as deferred
                    self._hub.learner.record_observation(
                        conditions={"action": "analyze_image"},
                        action="defer",
                        success=True,
                    )
                    raise RuntimeError("Request throttled due to high load")

            result = await self._base_provider.analyze_image(image_data, include_description)

            return result

        except Exception as e:
            error_occurred = True
            self._error_count += 1

            # Record for learning
            self._hub.learner.record_observation(
                conditions={"action": "analyze_image"},
                action="execute",
                success=False,
            )

            raise

        finally:
            latency_ms = (time.time() - start_time) * 1000
            self._request_count += 1
            self._total_latency += latency_ms

            # Record metrics
            with self._lock:
                error_rate = (
                    self._error_count / self._request_count if self._request_count > 0 else 0
                )
                avg_latency = (
                    self._total_latency / self._request_count if self._request_count > 0 else 0
                )

            metrics = LoadMetrics(
                cpu_usage=50.0,  # Placeholder - would come from system monitoring
                memory_usage=50.0,
                request_rate=1.0,
                error_rate=error_rate,
                latency_p50=avg_latency * 0.5,
                latency_p99=avg_latency * 1.5,
                queue_depth=0,
            )
            self._hub.process_metrics(metrics)

            if not error_occurred:
                self._hub.learner.record_observation(
                    conditions={"action": "analyze_image"},
                    action="execute",
                    success=True,
                )


# ========================
# Factory Functions
# ========================


def create_automation_config(
    decision_cooldown_seconds: int = 60,
    tuning_enabled: bool = True,
    prediction_enabled: bool = True,
    auto_remediation_enabled: bool = True,
    learning_enabled: bool = True,
    max_concurrent_tasks: int = 10,
    **kwargs: Any,
) -> AutomationConfig:
    """Create an automation configuration."""
    return AutomationConfig(
        decision_cooldown_seconds=decision_cooldown_seconds,
        tuning_enabled=tuning_enabled,
        prediction_enabled=prediction_enabled,
        auto_remediation_enabled=auto_remediation_enabled,
        learning_enabled=learning_enabled,
        max_concurrent_tasks=max_concurrent_tasks,
        **kwargs,
    )


def create_intelligent_automation_hub(
    decision_cooldown_seconds: int = 60,
    max_concurrent_tasks: int = 10,
    **kwargs: Any,
) -> IntelligentAutomationHub:
    """Create an intelligent automation hub."""
    config = create_automation_config(
        decision_cooldown_seconds=decision_cooldown_seconds,
        max_concurrent_tasks=max_concurrent_tasks,
        **kwargs,
    )
    return IntelligentAutomationHub(config)


def create_decision_rule(
    name: str,
    condition: Callable[[Dict[str, Any]], bool],
    decision_type: DecisionType,
    parameters_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    priority: int = 0,
    description: str = "",
) -> DecisionRule:
    """Create a decision rule."""
    return DecisionRule(
        rule_id=str(uuid.uuid4()),
        name=name,
        condition=condition,
        decision_type=decision_type,
        parameters_fn=parameters_fn,
        priority=priority,
        description=description,
    )


def create_load_metrics(
    cpu_usage: float,
    memory_usage: float,
    request_rate: float = 0.0,
    error_rate: float = 0.0,
    latency_p50: float = 0.0,
    latency_p99: float = 0.0,
    queue_depth: int = 0,
) -> LoadMetrics:
    """Create load metrics."""
    return LoadMetrics(
        cpu_usage=cpu_usage,
        memory_usage=memory_usage,
        request_rate=request_rate,
        error_rate=error_rate,
        latency_p50=latency_p50,
        latency_p99=latency_p99,
        queue_depth=queue_depth,
    )


def create_automated_provider(
    base_provider: VisionProvider,
    hub: Optional[IntelligentAutomationHub] = None,
) -> AutomatedVisionProvider:
    """Create an automated vision provider."""
    return AutomatedVisionProvider(base_provider, hub)
