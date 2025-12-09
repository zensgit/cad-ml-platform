# Phase 23: Intelligent Automation & Self-Optimization

## Overview

Phase 23 introduces a comprehensive intelligent automation system through the `intelligent_automation.py` module. This central hub integrates automated decision making, self-tuning parameter optimization, intelligent task scheduling, adaptive load management, performance prediction, automatic remediation, and pattern learning into a unified self-optimizing platform.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   IntelligentAutomationHub                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │  Decision   │ │   Self      │ │ Intelligent │ │    Load     │   │
│  │   Engine    │ │   Tuner     │ │  Scheduler  │ │   Manager   │   │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘   │
│         │               │               │               │           │
│  ┌──────┴───────────────┴───────────────┴───────────────┴──────┐   │
│  │                  Unified Automation Pipeline                  │   │
│  └──────┬───────────────┬───────────────┬───────────────┬──────┘   │
│         │               │               │               │           │
│  ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐                   │
│  │ Performance │ │    Auto     │ │   Pattern   │                   │
│  │  Predictor  │ │ Remediation │ │   Learner   │                   │
│  └─────────────┘ └─────────────┘ └─────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AutomatedVisionProvider                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Base VisionProvider + Load Shedding + Pattern Learning      │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Enums

| Enum | Values | Description |
|------|--------|-------------|
| `DecisionType` | scale_up, scale_down, route_traffic, failover, optimize, alert, remediate, defer | Types of automated decisions |
| `DecisionConfidence` | high, medium, low, uncertain | Confidence levels for decisions |
| `TuningStrategy` | gradient_descent, bayesian, genetic, grid_search, random_search, adaptive | Parameter tuning strategies |
| `TuningStatus` | idle, exploring, exploiting, converged, failed | Status of tuning operations |
| `SchedulerPriority` | critical, high, normal, low, background | Task scheduler priority levels |
| `TaskState` | pending, scheduled, running, completed, failed, cancelled, deferred | Scheduled task states |
| `LoadLevel` | critical, high, moderate, low, idle | System load levels |
| `RemediationAction` | restart, scale, failover, throttle, clear_cache, rollback, notify, custom | Types of remediation actions |
| `PredictionType` | load, latency, error_rate, resource_usage, throughput, cost | Types of predictions |
| `LearningMode` | online, batch, reinforcement, supervised | Learning modes for the system |

### 2. Dataclasses

#### Decision
```python
@dataclass
class Decision:
    decision_id: str
    decision_type: DecisionType
    confidence: DecisionConfidence
    rationale: str
    parameters: Dict[str, Any]
    timestamp: datetime
    executed: bool
    outcome: Optional[str]
    feedback_score: Optional[float]

    def to_dict() -> Dict[str, Any]
```

#### DecisionRule
```python
@dataclass
class DecisionRule:
    rule_id: str
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    decision_type: DecisionType
    parameters_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    priority: int
    enabled: bool
    description: str
```

#### TuningParameter
```python
@dataclass
class TuningParameter:
    name: str
    current_value: float
    min_value: float
    max_value: float
    step_size: float
    best_value: Optional[float]
    best_score: Optional[float]
    history: List[Tuple[float, float]]

    def get_next_value(strategy: TuningStrategy) -> float
```

#### TuningSession
```python
@dataclass
class TuningSession:
    session_id: str
    parameters: Dict[str, TuningParameter]
    strategy: TuningStrategy
    status: TuningStatus
    objective_fn: Optional[Callable]
    iterations: int
    max_iterations: int
    convergence_threshold: float
    started_at: datetime
    completed_at: Optional[datetime]
```

#### ScheduledTask
```python
@dataclass
class ScheduledTask:
    task_id: str
    name: str
    priority: SchedulerPriority
    state: TaskState
    execute_fn: Callable[[], Any]
    scheduled_time: datetime
    deadline: Optional[datetime]
    resource_requirements: Dict[str, float]
    dependencies: List[str]
    retries: int
    max_retries: int
    result: Optional[Any]
    error: Optional[str]
```

#### ResourcePool
```python
@dataclass
class ResourcePool:
    cpu: float
    memory: float
    io_bandwidth: float
    network: float
    custom: Dict[str, float]

    def can_allocate(requirements: Dict[str, float]) -> bool
    def allocate(requirements: Dict[str, float]) -> None
    def release(requirements: Dict[str, float]) -> None
```

#### LoadMetrics
```python
@dataclass
class LoadMetrics:
    cpu_usage: float
    memory_usage: float
    request_rate: float
    error_rate: float
    latency_p50: float
    latency_p99: float
    queue_depth: int
    timestamp: datetime

    def get_load_level() -> LoadLevel
```

#### Remediation
```python
@dataclass
class Remediation:
    remediation_id: str
    action: RemediationAction
    target: str
    reason: str
    parameters: Dict[str, Any]
    timestamp: datetime
    success: bool
    result: Optional[str]
    duration_ms: float
```

#### Prediction
```python
@dataclass
class Prediction:
    prediction_id: str
    prediction_type: PredictionType
    target_time: datetime
    predicted_value: float
    confidence_interval: Tuple[float, float]
    confidence: float
    model_version: str
    timestamp: datetime
    actual_value: Optional[float]
```

#### LearningPattern
```python
@dataclass
class LearningPattern:
    pattern_id: str
    name: str
    conditions: Dict[str, Any]
    optimal_action: str
    success_rate: float
    sample_count: int
    last_updated: datetime
```

#### AutomationConfig
```python
@dataclass
class AutomationConfig:
    decision_cooldown_seconds: int = 60
    tuning_enabled: bool = True
    prediction_enabled: bool = True
    auto_remediation_enabled: bool = True
    learning_enabled: bool = True
    max_concurrent_tasks: int = 10
    load_threshold_critical: float = 0.9
    load_threshold_high: float = 0.75
    prediction_horizon_minutes: int = 30
```

### 3. Core Classes

#### DecisionEngine
```python
class DecisionEngine:
    def add_rule(rule: DecisionRule) -> None
    def remove_rule(rule_id: str) -> bool
    def get_rule(rule_id: str) -> Optional[DecisionRule]
    def list_rules() -> List[DecisionRule]
    def evaluate(context: Dict[str, Any]) -> Optional[Decision]
    def record_feedback(decision_id: str, outcome: str, score: float) -> Optional[Decision]
    def get_decision_history(limit: int = 100) -> List[Decision]
    def get_decision_stats() -> Dict[str, Any]
```

#### SelfTuner
```python
class SelfTuner:
    def create_session(parameters: Dict[str, Tuple[float, float, float]],
                       strategy: TuningStrategy = TuningStrategy.ADAPTIVE,
                       objective_fn: Callable = None,
                       max_iterations: int = 100) -> TuningSession
    def get_session(session_id: str) -> Optional[TuningSession]
    def start_session(session_id: str) -> bool
    def step(session_id: str) -> Optional[Dict[str, float]]
    def record_result(session_id: str, values: Dict[str, float], score: float) -> bool
    def get_best_parameters(session_id: str) -> Optional[Dict[str, float]]
    def list_sessions() -> List[TuningSession]
```

#### IntelligentScheduler
```python
class IntelligentScheduler:
    def schedule_task(name: str, execute_fn: Callable,
                      priority: SchedulerPriority = SchedulerPriority.NORMAL,
                      scheduled_time: datetime = None,
                      deadline: datetime = None,
                      resource_requirements: Dict[str, float] = None,
                      dependencies: List[str] = None) -> ScheduledTask
    def get_task(task_id: str) -> Optional[ScheduledTask]
    def cancel_task(task_id: str) -> bool
    def get_next_task() -> Optional[ScheduledTask]
    def start_task(task_id: str) -> bool
    def complete_task(task_id: str, result: Any = None, error: str = None) -> bool
    def get_queue_stats() -> Dict[str, Any]
    def list_tasks(state: TaskState = None, limit: int = 100) -> List[ScheduledTask]
```

#### LoadManager
```python
class LoadManager:
    def record_metrics(metrics: LoadMetrics) -> LoadLevel
    def register_handler(level: LoadLevel, handler: Callable) -> None
    def get_current_load() -> Optional[LoadLevel]
    def get_average_metrics(window_seconds: int = 300) -> Optional[LoadMetrics]
    def get_metrics_history(limit: int = 100) -> List[LoadMetrics]
    def should_shed_load() -> bool
    def get_throttle_percentage() -> float
```

#### PerformancePredictor
```python
class PerformancePredictor:
    def record_observation(prediction_type: PredictionType, value: float) -> None
    def predict(prediction_type: PredictionType,
                horizon_minutes: int = None) -> Optional[Prediction]
    def validate_prediction(prediction_id: str, actual_value: float) -> Optional[Prediction]
    def get_prediction_accuracy(prediction_type: PredictionType = None) -> Dict[str, float]
    def get_predictions(limit: int = 100) -> List[Prediction]
```

#### AutoRemediation
```python
class AutoRemediation:
    def register_handler(action: RemediationAction,
                         handler: Callable[[str, Dict[str, Any]], bool]) -> None
    def execute_remediation(action: RemediationAction, target: str,
                            reason: str, parameters: Dict = None) -> Remediation
    def get_remediation_history(limit: int = 100, success_only: bool = False) -> List[Remediation]
    def get_remediation_stats() -> Dict[str, Any]
```

#### PatternLearner
```python
class PatternLearner:
    def record_observation(conditions: Dict[str, Any], action: str, success: bool) -> None
    def get_recommended_action(conditions: Dict[str, Any]) -> Optional[Tuple[str, float]]
    def get_patterns(min_samples: int = 5) -> List[LearningPattern]
    def get_learning_stats() -> Dict[str, Any]
```

#### IntelligentAutomationHub
```python
class IntelligentAutomationHub:
    @property decision_engine: DecisionEngine
    @property self_tuner: SelfTuner
    @property scheduler: IntelligentScheduler
    @property load_manager: LoadManager
    @property predictor: PerformancePredictor
    @property remediation: AutoRemediation
    @property learner: PatternLearner

    def process_metrics(metrics: LoadMetrics) -> Dict[str, Any]
    def get_automation_summary() -> Dict[str, Any]
```

### 4. Vision Provider Integration

#### AutomatedVisionProvider
```python
class AutomatedVisionProvider(VisionProvider):
    def __init__(base_provider: VisionProvider, hub: IntelligentAutomationHub)

    @property provider_name: str  # "automated_{base_provider_name}"

    async def analyze_image(image_data: bytes, include_description: bool) -> VisionDescription
```

Features:
- Automatic load shedding under high load
- Request throttling based on system metrics
- Pattern learning from request outcomes
- Metrics recording for predictions
- Integration with remediation system

### 5. Factory Functions

```python
def create_automation_config(
    decision_cooldown_seconds: int = 60,
    tuning_enabled: bool = True,
    prediction_enabled: bool = True,
    auto_remediation_enabled: bool = True,
    learning_enabled: bool = True,
    max_concurrent_tasks: int = 10,
    **kwargs
) -> AutomationConfig

def create_intelligent_automation_hub(
    decision_cooldown_seconds: int = 60,
    max_concurrent_tasks: int = 10,
    **kwargs
) -> IntelligentAutomationHub

def create_decision_rule(
    name: str,
    condition: Callable[[Dict[str, Any]], bool],
    decision_type: DecisionType,
    parameters_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    priority: int = 0,
    description: str = ""
) -> DecisionRule

def create_load_metrics(
    cpu_usage: float,
    memory_usage: float,
    request_rate: float = 0.0,
    error_rate: float = 0.0,
    latency_p50: float = 0.0,
    latency_p99: float = 0.0,
    queue_depth: int = 0
) -> LoadMetrics

def create_automated_provider(
    base_provider: VisionProvider,
    hub: IntelligentAutomationHub = None
) -> AutomatedVisionProvider
```

## Usage Examples

### Automated Decision Making
```python
from src.core.vision import (
    create_intelligent_automation_hub,
    create_decision_rule,
    DecisionType,
)

hub = create_intelligent_automation_hub()

# Add decision rule for high CPU
rule = create_decision_rule(
    name="High CPU Scale",
    condition=lambda ctx: ctx.get("cpu_usage", 0) > 80,
    decision_type=DecisionType.SCALE_UP,
    parameters_fn=lambda ctx: {"instances": 2},
    priority=10,
    description="Scale up when CPU > 80%"
)
hub.decision_engine.add_rule(rule)

# Evaluate context
context = {"cpu_usage": 90, "memory_usage": 70}
decision = hub.decision_engine.evaluate(context)
if decision:
    print(f"Decision: {decision.decision_type}, Params: {decision.parameters}")
```

### Self-Tuning Parameters
```python
from src.core.vision import create_intelligent_automation_hub, TuningStrategy

hub = create_intelligent_automation_hub()

# Create tuning session
session = hub.self_tuner.create_session(
    parameters={
        "learning_rate": (0.001, 0.1, 0.01),
        "batch_size": (16, 128, 32),
    },
    strategy=TuningStrategy.ADAPTIVE,
    max_iterations=100,
)

# Start tuning
hub.self_tuner.start_session(session.session_id)

# Iterate
for _ in range(50):
    values = hub.self_tuner.step(session.session_id)
    if values is None:
        break
    # Evaluate performance with these values
    score = evaluate_model(**values)
    hub.self_tuner.record_result(session.session_id, values, score)

# Get best parameters
best = hub.self_tuner.get_best_parameters(session.session_id)
print(f"Best parameters: {best}")
```

### Intelligent Task Scheduling
```python
from src.core.vision import create_intelligent_automation_hub, SchedulerPriority

hub = create_intelligent_automation_hub()

# Schedule tasks with dependencies
task1 = hub.scheduler.schedule_task(
    name="Data Preprocessing",
    execute_fn=lambda: preprocess_data(),
    priority=SchedulerPriority.HIGH,
    resource_requirements={"cpu": 30.0, "memory": 20.0},
)

task2 = hub.scheduler.schedule_task(
    name="Model Training",
    execute_fn=lambda: train_model(),
    priority=SchedulerPriority.HIGH,
    dependencies=[task1.task_id],
    resource_requirements={"cpu": 80.0, "memory": 50.0},
)

# Execute tasks
while True:
    task = hub.scheduler.get_next_task()
    if task is None:
        break
    hub.scheduler.start_task(task.task_id)
    result = task.execute_fn()
    hub.scheduler.complete_task(task.task_id, result=result)
```

### Load Management
```python
from src.core.vision import (
    create_intelligent_automation_hub,
    create_load_metrics,
    LoadLevel,
)

hub = create_intelligent_automation_hub()

# Register load handlers
def handle_critical_load(metrics):
    print(f"CRITICAL: CPU={metrics.cpu_usage}%, taking action...")

hub.load_manager.register_handler(LoadLevel.CRITICAL, handle_critical_load)

# Record metrics
metrics = create_load_metrics(
    cpu_usage=95.0,
    memory_usage=90.0,
    request_rate=1000.0,
    error_rate=0.05,
    latency_p99=500.0,
)
level = hub.load_manager.record_metrics(metrics)

# Check if load shedding needed
if hub.load_manager.should_shed_load():
    throttle = hub.load_manager.get_throttle_percentage()
    print(f"Load shedding active: {throttle}% throttle")
```

### Performance Prediction
```python
from src.core.vision import create_intelligent_automation_hub, PredictionType

hub = create_intelligent_automation_hub()

# Record historical observations
for i in range(100):
    hub.predictor.record_observation(PredictionType.LOAD, 50.0 + i * 0.5)

# Make prediction
prediction = hub.predictor.predict(PredictionType.LOAD, horizon_minutes=30)
if prediction:
    print(f"Predicted load in 30 min: {prediction.predicted_value:.1f}%")
    print(f"Confidence interval: {prediction.confidence_interval}")
```

### Auto-Remediation
```python
from src.core.vision import create_intelligent_automation_hub, RemediationAction

hub = create_intelligent_automation_hub()

# Register remediation handlers
def restart_service(target, params):
    print(f"Restarting {target}...")
    return True

hub.remediation.register_handler(RemediationAction.RESTART, restart_service)

# Execute remediation
result = hub.remediation.execute_remediation(
    action=RemediationAction.RESTART,
    target="vision-service",
    reason="High error rate detected",
    parameters={"graceful": True},
)
print(f"Remediation success: {result.success}")
```

### Pattern Learning
```python
from src.core.vision import create_intelligent_automation_hub

hub = create_intelligent_automation_hub()

# Learn from operations
for _ in range(20):
    hub.learner.record_observation(
        conditions={"load": "high", "time": "peak"},
        action="scale_up",
        success=True,
    )

# Get recommendation
recommendation = hub.learner.get_recommended_action(
    {"load": "high", "time": "peak"}
)
if recommendation:
    action, confidence = recommendation
    print(f"Recommended: {action} (confidence: {confidence:.2f})")
```

### Automated Vision Provider
```python
from src.core.vision import (
    create_automated_provider,
    create_intelligent_automation_hub,
)

hub = create_intelligent_automation_hub()
provider = create_automated_provider(my_base_provider, hub)

# Analyze with automation
result = await provider.analyze_image(image_data)

# Get automation summary
summary = hub.get_automation_summary()
print(f"Current load: {summary['current_load']}")
print(f"Decisions made: {summary['decision_stats']['total_decisions']}")
```

## Test Coverage

| Test Class | Tests | Coverage |
|------------|-------|----------|
| TestAutomationEnums | 9 | All enum values |
| TestAutomationDataclasses | 9 | All dataclass creation and methods |
| TestDecisionEngine | 7 | Rule CRUD and evaluation |
| TestSelfTuner | 5 | Session management and tuning |
| TestIntelligentScheduler | 6 | Task scheduling and execution |
| TestLoadManager | 6 | Metrics recording and load detection |
| TestPerformancePredictor | 6 | Prediction and validation |
| TestAutoRemediation | 5 | Handler registration and execution |
| TestPatternLearner | 4 | Observation and recommendations |
| TestIntelligentAutomationHub | 4 | Hub integration |
| TestAutomatedVisionProvider | 3 | Provider integration |
| TestFactoryFunctions | 5 | All factory functions |
| TestAutomationIntegration | 4 | End-to-end scenarios |

**Total: 73 tests**

## Integration with Existing Modules

Phase 23 complements existing modules:

| Existing Module | Phase 23 Enhancement |
|----------------|---------------------|
| `auto_scaling.py` | DecisionEngine adds rule-based scaling decisions |
| `self_healing.py` | AutoRemediation adds structured remediation |
| `predictive_analytics.py` | PerformancePredictor adds time-series forecasting |
| `intelligent_routing.py` | LoadManager adds adaptive load balancing |
| `workflow_engine.py` | IntelligentScheduler adds priority-based scheduling |

## Performance Considerations

- **Thread Safety**: All components use `threading.RLock()` for concurrent access
- **Decision Cooldown**: Configurable cooldown to prevent decision flooding
- **Memory Management**: History limits on metrics, predictions, and observations
- **Efficient Prediction**: Linear regression for fast predictions
- **Pattern Hashing**: MD5 hash for efficient pattern lookup

## Automation Features

| Feature | Implementation |
|---------|---------------|
| Decision Making | Rule-based with priority and cooldown |
| Parameter Tuning | Multiple strategies (random, grid, gradient, adaptive) |
| Task Scheduling | Priority and resource-aware scheduling |
| Load Management | Metric recording with level handlers |
| Performance Prediction | Linear regression with confidence intervals |
| Auto-Remediation | Handler-based remediation execution |
| Pattern Learning | Exponential moving average for success rates |

## Dependencies

- Standard library only (no external dependencies)
- Uses `threading`, `uuid`, `statistics`, `math`, `asyncio`
- Integrates with `VisionProvider` base class

## File Structure

```
src/core/vision/
├── intelligent_automation.py  # Phase 23 implementation
├── __init__.py                # Updated with Phase 23 exports
└── ...

tests/unit/
├── test_vision_phase23.py     # 73 comprehensive tests
└── ...

docs/
├── VISION_PHASE23_DESIGN.md   # This document
└── ...
```

## Summary

Phase 23 provides a unified intelligent automation platform that:
- Automates decisions based on configurable rules
- Self-tunes parameters using multiple optimization strategies
- Schedules tasks with priority, dependencies, and resource awareness
- Manages load with automatic shedding and throttling
- Predicts performance metrics for proactive scaling
- Executes automatic remediation actions
- Learns from operational patterns for better recommendations
- Wraps Vision providers with automation capabilities
