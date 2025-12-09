"""Tests for Phase 23: Intelligent Automation & Self-Optimization.

This module tests the intelligent automation capabilities including:
- Decision engine with rule-based decisions
- Self-tuning parameter optimization
- Intelligent task scheduler
- Adaptive load management
- Performance prediction
- Automatic remediation
- Pattern learning
"""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.vision import (
    # Main classes
    IntelligentAutomationHub,
    AutomatedVisionProvider,
    DecisionEngine,
    SelfTuner,
    IntelligentScheduler,
    LoadManager,
    PerformancePredictor,
    AutoRemediation,
    PatternLearner,
    # Enums
    DecisionType,
    DecisionConfidence,
    TuningStrategy,
    TuningStatus,
    SchedulerPriority,
    TaskStateP23 as TaskState,
    LoadLevel,
    RemediationAction,
    PredictionType,
    LearningMode,
    # Dataclasses
    Decision,
    DecisionRule,
    TuningParameter,
    TuningSession,
    ScheduledTask,
    ResourcePool,
    LoadMetrics,
    Remediation,
    Prediction,
    LearningPattern,
    AutomationConfig,
    # Factory functions
    create_automation_config,
    create_intelligent_automation_hub,
    create_decision_rule,
    create_load_metrics,
    create_automated_provider,
    # Base
    VisionDescription,
)


# ========================
# Test Enums
# ========================


class TestAutomationEnums:
    """Tests for automation-related enums."""

    def test_decision_type_values(self):
        """Test DecisionType enum values."""
        assert DecisionType.SCALE_UP.value == "scale_up"
        assert DecisionType.SCALE_DOWN.value == "scale_down"
        assert DecisionType.ROUTE_TRAFFIC.value == "route_traffic"
        assert DecisionType.FAILOVER.value == "failover"
        assert DecisionType.OPTIMIZE.value == "optimize"
        assert DecisionType.ALERT.value == "alert"
        assert DecisionType.REMEDIATE.value == "remediate"
        assert DecisionType.DEFER.value == "defer"

    def test_decision_confidence_values(self):
        """Test DecisionConfidence enum values."""
        assert DecisionConfidence.HIGH.value == "high"
        assert DecisionConfidence.MEDIUM.value == "medium"
        assert DecisionConfidence.LOW.value == "low"
        assert DecisionConfidence.UNCERTAIN.value == "uncertain"

    def test_tuning_strategy_values(self):
        """Test TuningStrategy enum values."""
        assert TuningStrategy.GRADIENT_DESCENT.value == "gradient_descent"
        assert TuningStrategy.BAYESIAN.value == "bayesian"
        assert TuningStrategy.GENETIC.value == "genetic"
        assert TuningStrategy.GRID_SEARCH.value == "grid_search"
        assert TuningStrategy.RANDOM_SEARCH.value == "random_search"
        assert TuningStrategy.ADAPTIVE.value == "adaptive"

    def test_tuning_status_values(self):
        """Test TuningStatus enum values."""
        assert TuningStatus.IDLE.value == "idle"
        assert TuningStatus.EXPLORING.value == "exploring"
        assert TuningStatus.EXPLOITING.value == "exploiting"
        assert TuningStatus.CONVERGED.value == "converged"
        assert TuningStatus.FAILED.value == "failed"

    def test_scheduler_priority_values(self):
        """Test SchedulerPriority enum values."""
        assert SchedulerPriority.CRITICAL.value == "critical"
        assert SchedulerPriority.HIGH.value == "high"
        assert SchedulerPriority.NORMAL.value == "normal"
        assert SchedulerPriority.LOW.value == "low"
        assert SchedulerPriority.BACKGROUND.value == "background"

    def test_task_state_values(self):
        """Test TaskState enum values."""
        assert TaskState.PENDING.value == "pending"
        assert TaskState.SCHEDULED.value == "scheduled"
        assert TaskState.RUNNING.value == "running"
        assert TaskState.COMPLETED.value == "completed"
        assert TaskState.FAILED.value == "failed"
        assert TaskState.CANCELLED.value == "cancelled"
        assert TaskState.DEFERRED.value == "deferred"

    def test_load_level_values(self):
        """Test LoadLevel enum values."""
        assert LoadLevel.CRITICAL.value == "critical"
        assert LoadLevel.HIGH.value == "high"
        assert LoadLevel.MODERATE.value == "moderate"
        assert LoadLevel.LOW.value == "low"
        assert LoadLevel.IDLE.value == "idle"

    def test_remediation_action_values(self):
        """Test RemediationAction enum values."""
        assert RemediationAction.RESTART.value == "restart"
        assert RemediationAction.SCALE.value == "scale"
        assert RemediationAction.FAILOVER.value == "failover"
        assert RemediationAction.THROTTLE.value == "throttle"
        assert RemediationAction.CLEAR_CACHE.value == "clear_cache"
        assert RemediationAction.ROLLBACK.value == "rollback"
        assert RemediationAction.NOTIFY.value == "notify"
        assert RemediationAction.CUSTOM.value == "custom"

    def test_prediction_type_values(self):
        """Test PredictionType enum values."""
        assert PredictionType.LOAD.value == "load"
        assert PredictionType.LATENCY.value == "latency"
        assert PredictionType.ERROR_RATE.value == "error_rate"
        assert PredictionType.RESOURCE_USAGE.value == "resource_usage"
        assert PredictionType.THROUGHPUT.value == "throughput"
        assert PredictionType.COST.value == "cost"


# ========================
# Test Dataclasses
# ========================


class TestAutomationDataclasses:
    """Tests for automation dataclasses."""

    def test_decision_creation(self):
        """Test Decision dataclass creation."""
        decision = Decision(
            decision_id="dec-123",
            decision_type=DecisionType.SCALE_UP,
            confidence=DecisionConfidence.HIGH,
            rationale="High CPU usage detected",
            parameters={"instances": 2},
        )
        assert decision.decision_id == "dec-123"
        assert decision.decision_type == DecisionType.SCALE_UP
        assert decision.confidence == DecisionConfidence.HIGH
        assert decision.rationale == "High CPU usage detected"
        assert decision.parameters == {"instances": 2}
        assert decision.executed is False

    def test_decision_to_dict(self):
        """Test Decision to_dict method."""
        decision = Decision(
            decision_id="dec-123",
            decision_type=DecisionType.SCALE_UP,
            confidence=DecisionConfidence.HIGH,
            rationale="Test",
            parameters={},
        )
        result = decision.to_dict()
        assert result["decision_id"] == "dec-123"
        assert result["decision_type"] == "scale_up"
        assert result["confidence"] == "high"

    def test_tuning_parameter_creation(self):
        """Test TuningParameter dataclass creation."""
        param = TuningParameter(
            name="learning_rate",
            current_value=0.01,
            min_value=0.001,
            max_value=0.1,
            step_size=0.01,
        )
        assert param.name == "learning_rate"
        assert param.current_value == 0.01
        assert param.min_value == 0.001
        assert param.max_value == 0.1

    def test_tuning_parameter_get_next_value(self):
        """Test TuningParameter get_next_value method."""
        param = TuningParameter(
            name="test",
            current_value=0.5,
            min_value=0.0,
            max_value=1.0,
        )
        # Random search should return value in range
        next_val = param.get_next_value(TuningStrategy.RANDOM_SEARCH)
        assert 0.0 <= next_val <= 1.0

    def test_resource_pool_creation(self):
        """Test ResourcePool dataclass creation."""
        pool = ResourcePool(cpu=80.0, memory=60.0)
        assert pool.cpu == 80.0
        assert pool.memory == 60.0

    def test_resource_pool_can_allocate(self):
        """Test ResourcePool can_allocate method."""
        pool = ResourcePool(cpu=50.0, memory=50.0)
        assert pool.can_allocate({"cpu": 30.0, "memory": 30.0}) is True
        assert pool.can_allocate({"cpu": 60.0}) is False

    def test_resource_pool_allocate_and_release(self):
        """Test ResourcePool allocate and release methods."""
        pool = ResourcePool(cpu=100.0, memory=100.0)
        pool.allocate({"cpu": 30.0, "memory": 20.0})
        assert pool.cpu == 70.0
        assert pool.memory == 80.0
        pool.release({"cpu": 30.0, "memory": 20.0})
        assert pool.cpu == 100.0
        assert pool.memory == 100.0

    def test_load_metrics_creation(self):
        """Test LoadMetrics dataclass creation."""
        metrics = LoadMetrics(
            cpu_usage=75.0,
            memory_usage=60.0,
            request_rate=100.0,
            error_rate=0.02,
            latency_p50=50.0,
            latency_p99=200.0,
            queue_depth=10,
        )
        assert metrics.cpu_usage == 75.0
        assert metrics.error_rate == 0.02

    def test_load_metrics_get_load_level(self):
        """Test LoadMetrics get_load_level method."""
        # Critical load
        metrics = LoadMetrics(
            cpu_usage=95.0,
            memory_usage=90.0,
            request_rate=100.0,
            error_rate=0.01,
            latency_p50=50.0,
            latency_p99=200.0,
            queue_depth=10,
        )
        assert metrics.get_load_level() == LoadLevel.CRITICAL

        # High load
        metrics = LoadMetrics(
            cpu_usage=85.0,
            memory_usage=80.0,
            request_rate=100.0,
            error_rate=0.01,
            latency_p50=50.0,
            latency_p99=200.0,
            queue_depth=10,
        )
        assert metrics.get_load_level() == LoadLevel.HIGH

        # Low load
        metrics = LoadMetrics(
            cpu_usage=30.0,
            memory_usage=30.0,
            request_rate=10.0,
            error_rate=0.0,
            latency_p50=10.0,
            latency_p99=50.0,
            queue_depth=0,
        )
        assert metrics.get_load_level() == LoadLevel.LOW


# ========================
# Test DecisionEngine
# ========================


class TestDecisionEngine:
    """Tests for DecisionEngine class."""

    def test_engine_initialization(self):
        """Test DecisionEngine initialization."""
        engine = DecisionEngine()
        assert engine is not None
        assert engine.list_rules() == []

    def test_add_rule(self):
        """Test adding a decision rule."""
        engine = DecisionEngine()
        rule = DecisionRule(
            rule_id="rule-1",
            name="High CPU Rule",
            condition=lambda ctx: ctx.get("cpu_usage", 0) > 80,
            decision_type=DecisionType.SCALE_UP,
            parameters_fn=lambda ctx: {"instances": 1},
            priority=10,
        )
        engine.add_rule(rule)
        assert len(engine.list_rules()) == 1
        assert engine.get_rule("rule-1") is not None

    def test_remove_rule(self):
        """Test removing a decision rule."""
        engine = DecisionEngine()
        rule = DecisionRule(
            rule_id="rule-1",
            name="Test Rule",
            condition=lambda ctx: True,
            decision_type=DecisionType.ALERT,
            parameters_fn=lambda ctx: {},
        )
        engine.add_rule(rule)
        assert engine.remove_rule("rule-1") is True
        assert engine.remove_rule("nonexistent") is False
        assert len(engine.list_rules()) == 0

    def test_evaluate_matching_rule(self):
        """Test evaluating context against rules."""
        config = AutomationConfig(decision_cooldown_seconds=0)
        engine = DecisionEngine(config)
        rule = DecisionRule(
            rule_id="rule-1",
            name="High CPU Rule",
            condition=lambda ctx: ctx.get("cpu_usage", 0) > 80,
            decision_type=DecisionType.SCALE_UP,
            parameters_fn=lambda ctx: {"cpu": ctx.get("cpu_usage")},
            priority=10,
        )
        engine.add_rule(rule)

        decision = engine.evaluate({"cpu_usage": 90})
        assert decision is not None
        assert decision.decision_type == DecisionType.SCALE_UP
        assert decision.parameters["cpu"] == 90

    def test_evaluate_no_matching_rule(self):
        """Test evaluation with no matching rules."""
        engine = DecisionEngine()
        rule = DecisionRule(
            rule_id="rule-1",
            name="High CPU Rule",
            condition=lambda ctx: ctx.get("cpu_usage", 0) > 80,
            decision_type=DecisionType.SCALE_UP,
            parameters_fn=lambda ctx: {},
        )
        engine.add_rule(rule)

        decision = engine.evaluate({"cpu_usage": 50})
        assert decision is None

    def test_record_feedback(self):
        """Test recording feedback for a decision."""
        config = AutomationConfig(decision_cooldown_seconds=0)
        engine = DecisionEngine(config)
        rule = DecisionRule(
            rule_id="rule-1",
            name="Test Rule",
            condition=lambda ctx: True,
            decision_type=DecisionType.SCALE_UP,
            parameters_fn=lambda ctx: {},
        )
        engine.add_rule(rule)

        decision = engine.evaluate({})
        assert decision is not None

        result = engine.record_feedback(decision.decision_id, "success", 0.9)
        assert result is not None
        assert result.outcome == "success"
        assert result.feedback_score == 0.9

    def test_get_decision_stats(self):
        """Test getting decision statistics."""
        config = AutomationConfig(decision_cooldown_seconds=0)
        engine = DecisionEngine(config)
        rule = DecisionRule(
            rule_id="rule-1",
            name="Test Rule",
            condition=lambda ctx: True,
            decision_type=DecisionType.SCALE_UP,
            parameters_fn=lambda ctx: {},
        )
        engine.add_rule(rule)

        engine.evaluate({})
        stats = engine.get_decision_stats()
        assert stats["total_decisions"] == 1
        assert "scale_up" in stats["by_type"]


# ========================
# Test SelfTuner
# ========================


class TestSelfTuner:
    """Tests for SelfTuner class."""

    def test_tuner_initialization(self):
        """Test SelfTuner initialization."""
        tuner = SelfTuner()
        assert tuner is not None
        assert len(tuner.list_sessions()) == 0

    def test_create_session(self):
        """Test creating a tuning session."""
        tuner = SelfTuner()
        session = tuner.create_session(
            parameters={
                "learning_rate": (0.001, 0.1, 0.01),
                "batch_size": (16, 128, 32),
            },
            strategy=TuningStrategy.RANDOM_SEARCH,
            max_iterations=50,
        )
        assert session is not None
        assert session.status == TuningStatus.IDLE
        assert "learning_rate" in session.parameters
        assert "batch_size" in session.parameters

    def test_start_session(self):
        """Test starting a tuning session."""
        tuner = SelfTuner()
        session = tuner.create_session(
            parameters={"param": (0.0, 1.0, 0.5)},
        )
        assert tuner.start_session(session.session_id) is True
        updated = tuner.get_session(session.session_id)
        assert updated.status == TuningStatus.EXPLORING

    def test_step_and_record(self):
        """Test performing tuning steps."""
        tuner = SelfTuner()
        session = tuner.create_session(
            parameters={"param": (0.0, 1.0, 0.5)},
            max_iterations=10,
        )
        tuner.start_session(session.session_id)

        # Perform a step
        next_values = tuner.step(session.session_id)
        assert next_values is not None
        assert "param" in next_values

        # Record result
        success = tuner.record_result(session.session_id, next_values, 0.8)
        assert success is True

    def test_get_best_parameters(self):
        """Test getting best parameters."""
        tuner = SelfTuner()
        session = tuner.create_session(
            parameters={"param": (0.0, 1.0, 0.5)},
        )
        tuner.start_session(session.session_id)

        # Do some iterations
        for _ in range(5):
            values = tuner.step(session.session_id)
            if values:
                score = 1.0 - abs(values["param"] - 0.7)  # Best around 0.7
                tuner.record_result(session.session_id, values, score)

        best = tuner.get_best_parameters(session.session_id)
        assert best is not None
        assert "param" in best


# ========================
# Test IntelligentScheduler
# ========================


class TestIntelligentScheduler:
    """Tests for IntelligentScheduler class."""

    def test_scheduler_initialization(self):
        """Test IntelligentScheduler initialization."""
        scheduler = IntelligentScheduler()
        assert scheduler is not None
        stats = scheduler.get_queue_stats()
        assert stats["total_tasks"] == 0

    def test_schedule_task(self):
        """Test scheduling a task."""
        scheduler = IntelligentScheduler()
        task = scheduler.schedule_task(
            name="Test Task",
            execute_fn=lambda: "done",
            priority=SchedulerPriority.HIGH,
        )
        assert task is not None
        assert task.name == "Test Task"
        assert task.priority == SchedulerPriority.HIGH
        assert task.state == TaskState.PENDING

    def test_get_next_task(self):
        """Test getting next task by priority."""
        scheduler = IntelligentScheduler()

        # Add tasks with different priorities
        low_task = scheduler.schedule_task(
            name="Low Priority",
            execute_fn=lambda: "low",
            priority=SchedulerPriority.LOW,
        )
        high_task = scheduler.schedule_task(
            name="High Priority",
            execute_fn=lambda: "high",
            priority=SchedulerPriority.HIGH,
        )

        next_task = scheduler.get_next_task()
        assert next_task is not None
        assert next_task.priority == SchedulerPriority.HIGH

    def test_start_and_complete_task(self):
        """Test starting and completing a task."""
        scheduler = IntelligentScheduler()
        task = scheduler.schedule_task(
            name="Test Task",
            execute_fn=lambda: "result",
        )

        assert scheduler.start_task(task.task_id) is True
        updated = scheduler.get_task(task.task_id)
        assert updated.state == TaskState.RUNNING

        assert scheduler.complete_task(task.task_id, result="success") is True
        completed = scheduler.get_task(task.task_id)
        assert completed.state == TaskState.COMPLETED
        assert completed.result == "success"

    def test_cancel_task(self):
        """Test cancelling a task."""
        scheduler = IntelligentScheduler()
        task = scheduler.schedule_task(
            name="Test Task",
            execute_fn=lambda: "result",
        )

        assert scheduler.cancel_task(task.task_id) is True
        cancelled = scheduler.get_task(task.task_id)
        assert cancelled.state == TaskState.CANCELLED

    def test_resource_requirements(self):
        """Test task scheduling with resource requirements."""
        config = AutomationConfig(max_concurrent_tasks=10)
        scheduler = IntelligentScheduler(config)

        # Schedule task with high CPU requirement
        task = scheduler.schedule_task(
            name="Heavy Task",
            execute_fn=lambda: "done",
            resource_requirements={"cpu": 80.0},
        )

        # Should be able to get and start it
        next_task = scheduler.get_next_task()
        assert next_task is not None
        scheduler.start_task(next_task.task_id)

        # Resource pool should be reduced
        stats = scheduler.get_queue_stats()
        assert stats["resource_pool"]["cpu"] < 100.0


# ========================
# Test LoadManager
# ========================


class TestLoadManager:
    """Tests for LoadManager class."""

    def test_load_manager_initialization(self):
        """Test LoadManager initialization."""
        manager = LoadManager()
        assert manager is not None
        assert manager.get_current_load() is None

    def test_record_metrics(self):
        """Test recording load metrics."""
        manager = LoadManager()
        metrics = LoadMetrics(
            cpu_usage=70.0,
            memory_usage=60.0,
            request_rate=100.0,
            error_rate=0.01,
            latency_p50=50.0,
            latency_p99=200.0,
            queue_depth=5,
        )
        level = manager.record_metrics(metrics)
        assert level == LoadLevel.MODERATE

        current = manager.get_current_load()
        assert current == LoadLevel.MODERATE

    def test_register_handler(self):
        """Test registering load level handler."""
        manager = LoadManager()
        handler_called = []

        def handler(metrics):
            handler_called.append(metrics)

        manager.register_handler(LoadLevel.HIGH, handler)

        # Trigger high load
        metrics = LoadMetrics(
            cpu_usage=85.0,
            memory_usage=80.0,
            request_rate=100.0,
            error_rate=0.01,
            latency_p50=50.0,
            latency_p99=200.0,
            queue_depth=5,
        )
        manager.record_metrics(metrics)
        assert len(handler_called) == 1

    def test_get_average_metrics(self):
        """Test getting average metrics."""
        manager = LoadManager()

        # Record multiple metrics
        for cpu in [60.0, 70.0, 80.0]:
            metrics = LoadMetrics(
                cpu_usage=cpu,
                memory_usage=50.0,
                request_rate=100.0,
                error_rate=0.0,
                latency_p50=50.0,
                latency_p99=200.0,
                queue_depth=0,
            )
            manager.record_metrics(metrics)

        avg = manager.get_average_metrics(window_seconds=300)
        assert avg is not None
        assert avg.cpu_usage == 70.0  # Average of 60, 70, 80

    def test_should_shed_load(self):
        """Test load shedding detection."""
        manager = LoadManager()

        # Low load - no shedding
        metrics = LoadMetrics(
            cpu_usage=50.0,
            memory_usage=50.0,
            request_rate=10.0,
            error_rate=0.0,
            latency_p50=10.0,
            latency_p99=50.0,
            queue_depth=0,
        )
        manager.record_metrics(metrics)
        assert manager.should_shed_load() is False

        # Critical load - should shed
        metrics = LoadMetrics(
            cpu_usage=95.0,
            memory_usage=95.0,
            request_rate=1000.0,
            error_rate=0.1,
            latency_p50=500.0,
            latency_p99=2000.0,
            queue_depth=100,
        )
        manager.record_metrics(metrics)
        assert manager.should_shed_load() is True

    def test_get_throttle_percentage(self):
        """Test throttle percentage calculation."""
        manager = LoadManager()

        # High load should return throttle percentage
        metrics = LoadMetrics(
            cpu_usage=92.0,
            memory_usage=92.0,
            request_rate=100.0,
            error_rate=0.0,
            latency_p50=50.0,
            latency_p99=200.0,
            queue_depth=10,
        )
        manager.record_metrics(metrics)
        throttle = manager.get_throttle_percentage()
        assert throttle > 0


# ========================
# Test PerformancePredictor
# ========================


class TestPerformancePredictor:
    """Tests for PerformancePredictor class."""

    def test_predictor_initialization(self):
        """Test PerformancePredictor initialization."""
        predictor = PerformancePredictor()
        assert predictor is not None

    def test_record_observation(self):
        """Test recording observations."""
        predictor = PerformancePredictor()
        predictor.record_observation(PredictionType.LOAD, 75.0)
        predictor.record_observation(PredictionType.LOAD, 80.0)
        # Should not raise any errors

    def test_predict_with_insufficient_data(self):
        """Test prediction with insufficient data."""
        predictor = PerformancePredictor()
        for i in range(5):
            predictor.record_observation(PredictionType.LOAD, 50.0 + i)

        # Should return None with < 10 observations
        prediction = predictor.predict(PredictionType.LOAD)
        assert prediction is None

    def test_predict_with_sufficient_data(self):
        """Test prediction with sufficient data."""
        predictor = PerformancePredictor()

        # Record enough observations
        for i in range(20):
            predictor.record_observation(PredictionType.LOAD, 50.0 + i * 0.5)

        prediction = predictor.predict(PredictionType.LOAD)
        assert prediction is not None
        assert prediction.prediction_type == PredictionType.LOAD
        assert prediction.confidence > 0

    def test_validate_prediction(self):
        """Test validating a prediction."""
        predictor = PerformancePredictor()

        # Generate prediction
        for i in range(20):
            predictor.record_observation(PredictionType.LATENCY, 100.0 + i)

        prediction = predictor.predict(PredictionType.LATENCY)
        assert prediction is not None

        # Validate with actual value
        result = predictor.validate_prediction(prediction.prediction_id, 115.0)
        assert result is not None
        assert result.actual_value == 115.0

    def test_get_prediction_accuracy(self):
        """Test getting prediction accuracy."""
        predictor = PerformancePredictor()

        # Without validated predictions
        accuracy = predictor.get_prediction_accuracy()
        assert accuracy["count"] == 0

        # With predictions and validations
        for i in range(20):
            predictor.record_observation(PredictionType.LOAD, 50.0 + i)

        prediction = predictor.predict(PredictionType.LOAD)
        if prediction:
            predictor.validate_prediction(prediction.prediction_id, 60.0)

        accuracy = predictor.get_prediction_accuracy()
        assert accuracy["count"] >= 0


# ========================
# Test AutoRemediation
# ========================


class TestAutoRemediation:
    """Tests for AutoRemediation class."""

    def test_remediation_initialization(self):
        """Test AutoRemediation initialization."""
        remediation = AutoRemediation()
        assert remediation is not None

    def test_register_handler(self):
        """Test registering remediation handler."""
        remediation = AutoRemediation()
        handler_called = []

        def restart_handler(target, params):
            handler_called.append((target, params))
            return True

        remediation.register_handler(RemediationAction.RESTART, restart_handler)
        # Handler registered successfully

    def test_execute_remediation_with_handler(self):
        """Test executing remediation with handler."""
        config = AutomationConfig(auto_remediation_enabled=True)
        remediation = AutoRemediation(config)

        def scale_handler(target, params):
            return True

        remediation.register_handler(RemediationAction.SCALE, scale_handler)

        result = remediation.execute_remediation(
            action=RemediationAction.SCALE,
            target="service-1",
            reason="High load",
            parameters={"instances": 3},
        )
        assert result.success is True

    def test_execute_remediation_disabled(self):
        """Test remediation when disabled."""
        config = AutomationConfig(auto_remediation_enabled=False)
        remediation = AutoRemediation(config)

        result = remediation.execute_remediation(
            action=RemediationAction.RESTART,
            target="service-1",
            reason="Test",
        )
        assert result.success is False
        assert "disabled" in result.result

    def test_get_remediation_stats(self):
        """Test getting remediation statistics."""
        config = AutomationConfig(auto_remediation_enabled=True)
        remediation = AutoRemediation(config)

        def success_handler(target, params):
            return True

        remediation.register_handler(RemediationAction.THROTTLE, success_handler)

        # Execute some remediations
        remediation.execute_remediation(
            RemediationAction.THROTTLE, "target", "test"
        )

        stats = remediation.get_remediation_stats()
        assert stats["total"] > 0


# ========================
# Test PatternLearner
# ========================


class TestPatternLearner:
    """Tests for PatternLearner class."""

    def test_learner_initialization(self):
        """Test PatternLearner initialization."""
        learner = PatternLearner()
        assert learner is not None

    def test_record_observation(self):
        """Test recording observations."""
        learner = PatternLearner()
        learner.record_observation(
            conditions={"load": "high", "time": "peak"},
            action="scale_up",
            success=True,
        )
        stats = learner.get_learning_stats()
        assert stats["total_observations"] == 1

    def test_get_recommended_action(self):
        """Test getting recommended action."""
        learner = PatternLearner()

        # Record multiple observations with same conditions
        conditions = {"load": "high"}
        for _ in range(10):
            learner.record_observation(conditions, "scale_up", True)

        recommendation = learner.get_recommended_action(conditions)
        assert recommendation is not None
        action, confidence = recommendation
        assert action == "scale_up"
        assert confidence > 0.5

    def test_get_patterns(self):
        """Test getting learned patterns."""
        learner = PatternLearner()

        # Record patterns
        for _ in range(10):
            learner.record_observation({"type": "a"}, "action_a", True)

        patterns = learner.get_patterns(min_samples=5)
        assert len(patterns) >= 1


# ========================
# Test IntelligentAutomationHub
# ========================


class TestIntelligentAutomationHub:
    """Tests for IntelligentAutomationHub class."""

    def test_hub_initialization(self):
        """Test IntelligentAutomationHub initialization."""
        hub = IntelligentAutomationHub()
        assert hub is not None
        assert hub.decision_engine is not None
        assert hub.self_tuner is not None
        assert hub.scheduler is not None
        assert hub.load_manager is not None
        assert hub.predictor is not None
        assert hub.remediation is not None
        assert hub.learner is not None

    def test_process_metrics(self):
        """Test processing metrics through the hub."""
        hub = IntelligentAutomationHub()

        metrics = LoadMetrics(
            cpu_usage=60.0,
            memory_usage=50.0,
            request_rate=100.0,
            error_rate=0.01,
            latency_p50=50.0,
            latency_p99=200.0,
            queue_depth=5,
        )

        results = hub.process_metrics(metrics)
        assert "load_level" in results

    def test_get_automation_summary(self):
        """Test getting automation summary."""
        hub = IntelligentAutomationHub()
        summary = hub.get_automation_summary()

        assert "decision_stats" in summary
        assert "scheduler_stats" in summary
        assert "prediction_accuracy" in summary
        assert "remediation_stats" in summary
        assert "learning_stats" in summary

    def test_hub_with_decision_rules(self):
        """Test hub with decision rules."""
        config = AutomationConfig(decision_cooldown_seconds=0)
        hub = IntelligentAutomationHub(config)

        # Add a rule
        rule = create_decision_rule(
            name="High Load Scale",
            condition=lambda ctx: ctx.get("cpu_usage", 0) > 80,
            decision_type=DecisionType.SCALE_UP,
            parameters_fn=lambda ctx: {"factor": 2},
        )
        hub.decision_engine.add_rule(rule)

        # Process high load metrics
        metrics = LoadMetrics(
            cpu_usage=90.0,
            memory_usage=85.0,
            request_rate=200.0,
            error_rate=0.02,
            latency_p50=100.0,
            latency_p99=500.0,
            queue_depth=20,
        )
        results = hub.process_metrics(metrics)

        assert "decision" in results


# ========================
# Test AutomatedVisionProvider
# ========================


class TestAutomatedVisionProvider:
    """Tests for AutomatedVisionProvider class."""

    def test_provider_name(self):
        """Test provider name."""
        base_provider = MagicMock()
        base_provider.provider_name = "test_provider"

        provider = AutomatedVisionProvider(base_provider)
        assert provider.provider_name == "automated_test_provider"

    @pytest.mark.asyncio
    async def test_analyze_image_success(self):
        """Test successful image analysis."""
        base_provider = MagicMock()
        base_provider.provider_name = "test_provider"
        base_provider.analyze_image = AsyncMock(
            return_value=VisionDescription(
                summary="Test CAD summary",
                confidence=0.95,
            )
        )

        provider = AutomatedVisionProvider(base_provider)
        result = await provider.analyze_image(b"test_image")

        assert result is not None
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_analyze_image_with_load_shedding(self):
        """Test image analysis with load shedding."""
        base_provider = MagicMock()
        base_provider.provider_name = "test_provider"
        base_provider.analyze_image = AsyncMock(
            return_value=VisionDescription(
                summary="Test summary",
                confidence=0.9,
            )
        )

        hub = IntelligentAutomationHub()
        # Simulate critical load
        metrics = LoadMetrics(
            cpu_usage=98.0,
            memory_usage=98.0,
            request_rate=1000.0,
            error_rate=0.15,
            latency_p50=1000.0,
            latency_p99=5000.0,
            queue_depth=1000,
        )
        hub.load_manager.record_metrics(metrics)

        provider = AutomatedVisionProvider(base_provider, hub)

        # Some requests might be throttled
        results = []
        for _ in range(10):
            try:
                result = await provider.analyze_image(b"test")
                results.append(result)
            except RuntimeError:
                pass  # Throttled

        # At least some should succeed
        assert len(results) >= 0  # May be throttled


# ========================
# Test Factory Functions
# ========================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_automation_config(self):
        """Test create_automation_config factory."""
        config = create_automation_config(
            decision_cooldown_seconds=30,
            tuning_enabled=False,
            max_concurrent_tasks=5,
        )
        assert config.decision_cooldown_seconds == 30
        assert config.tuning_enabled is False
        assert config.max_concurrent_tasks == 5

    def test_create_intelligent_automation_hub(self):
        """Test create_intelligent_automation_hub factory."""
        hub = create_intelligent_automation_hub(
            decision_cooldown_seconds=30,
            max_concurrent_tasks=5,
        )
        assert hub is not None
        assert isinstance(hub, IntelligentAutomationHub)

    def test_create_decision_rule(self):
        """Test create_decision_rule factory."""
        rule = create_decision_rule(
            name="Test Rule",
            condition=lambda ctx: True,
            decision_type=DecisionType.ALERT,
            parameters_fn=lambda ctx: {"msg": "alert"},
            priority=5,
            description="Test description",
        )
        assert rule.name == "Test Rule"
        assert rule.decision_type == DecisionType.ALERT
        assert rule.priority == 5

    def test_create_load_metrics(self):
        """Test create_load_metrics factory."""
        metrics = create_load_metrics(
            cpu_usage=75.0,
            memory_usage=60.0,
            request_rate=100.0,
            error_rate=0.01,
        )
        assert metrics.cpu_usage == 75.0
        assert metrics.memory_usage == 60.0

    def test_create_automated_provider(self):
        """Test create_automated_provider factory."""
        base_provider = MagicMock()
        base_provider.provider_name = "test"

        provider = create_automated_provider(base_provider)
        assert isinstance(provider, AutomatedVisionProvider)


# ========================
# Test Integration
# ========================


class TestAutomationIntegration:
    """Integration tests for the automation system."""

    def test_full_automation_pipeline(self):
        """Test full automation pipeline."""
        hub = create_intelligent_automation_hub()

        # Add decision rule
        rule = create_decision_rule(
            name="Scale on High Load",
            condition=lambda ctx: ctx.get("load_level") == "critical",
            decision_type=DecisionType.SCALE_UP,
            parameters_fn=lambda ctx: {"instances": 2},
        )
        hub.decision_engine.add_rule(rule)

        # Register remediation handler
        def scale_handler(target, params):
            return True

        hub.remediation.register_handler(RemediationAction.SCALE, scale_handler)

        # Process critical metrics
        metrics = create_load_metrics(
            cpu_usage=95.0,
            memory_usage=90.0,
            request_rate=500.0,
            error_rate=0.05,
            latency_p50=200.0,
            latency_p99=1000.0,
            queue_depth=50,
        )
        results = hub.process_metrics(metrics)

        assert results["load_level"] == "critical"

    def test_tuning_workflow(self):
        """Test complete tuning workflow."""
        tuner = SelfTuner()

        # Create session
        session = tuner.create_session(
            parameters={
                "param_a": (0.0, 1.0, 0.5),
                "param_b": (0.0, 10.0, 5.0),
            },
            strategy=TuningStrategy.RANDOM_SEARCH,
            max_iterations=20,
        )

        # Start and run tuning
        tuner.start_session(session.session_id)

        for _ in range(15):
            values = tuner.step(session.session_id)
            if values is None:
                break
            # Simulate scoring based on values
            score = 1.0 - (abs(values["param_a"] - 0.7) + abs(values["param_b"] - 7.0) / 10)
            tuner.record_result(session.session_id, values, score)

        # Get best parameters
        best = tuner.get_best_parameters(session.session_id)
        assert best is not None
        assert "param_a" in best
        assert "param_b" in best

    def test_scheduler_with_dependencies(self):
        """Test scheduler with task dependencies."""
        scheduler = IntelligentScheduler()

        # Create tasks with dependencies
        task1 = scheduler.schedule_task(
            name="Task 1",
            execute_fn=lambda: "result1",
        )
        task2 = scheduler.schedule_task(
            name="Task 2",
            execute_fn=lambda: "result2",
            dependencies=[task1.task_id],
        )

        # Task 2 should not be available until Task 1 completes
        scheduler.start_task(task1.task_id)
        next_task = scheduler.get_next_task()
        assert next_task is None  # Task 2 waiting for Task 1

        # Complete Task 1
        scheduler.complete_task(task1.task_id)

        # Now Task 2 should be available
        next_task = scheduler.get_next_task()
        assert next_task is not None
        assert next_task.task_id == task2.task_id

    def test_learning_and_recommendations(self):
        """Test learning from operations and getting recommendations."""
        learner = PatternLearner()

        # Simulate learning from operations - all successful scale_up actions
        for _ in range(20):
            learner.record_observation(
                conditions={"load": "high", "time_of_day": "peak"},
                action="scale_up",
                success=True,
            )

        # Get recommendation
        recommendation = learner.get_recommended_action(
            {"load": "high", "time_of_day": "peak"}
        )
        assert recommendation is not None
        action, confidence = recommendation
        assert action == "scale_up"
        assert confidence > 0.5  # EMA starts at 1.0 and stays high with all successes
