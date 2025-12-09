# Phase 21: Advanced Observability & Telemetry

## Overview

Phase 21 introduces a comprehensive observability orchestration system through the `observability_hub.py` module. This central hub integrates metrics collection, distributed tracing, health monitoring, alerting, SLO tracking, and anomaly detection into a unified observability platform.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ObservabilityHub                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │  Metrics    │ │   Trace     │ │   Health    │ │   Alert     │   │
│  │  Collector  │ │   Manager   │ │   Monitor   │ │   Manager   │   │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘   │
│         │               │               │               │           │
│  ┌──────┴───────────────┴───────────────┴───────────────┴──────┐   │
│  │                    Unified Data Pipeline                      │   │
│  └──────┬───────────────┬───────────────┬───────────────┬──────┘   │
│         │               │               │               │           │
│  ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐                   │
│  │    SLO      │ │   Anomaly   │ │  Dashboard  │                   │
│  │   Tracker   │ │  Detector   │ │    Data     │                   │
│  └─────────────┘ └─────────────┘ └─────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ObservableVisionProvider                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Base VisionProvider + Auto Metrics/Tracing/Alerting        │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Enums

| Enum | Values | Description |
|------|--------|-------------|
| `MetricType` | counter, gauge, histogram, summary, timer | Types of metrics |
| `HealthStatus` | healthy, degraded, unhealthy, unknown | Health check states |
| `AlertSeverity` | info, warning, error, critical | Alert priority levels |
| `AlertState` | pending, firing, resolved, suppressed | Alert lifecycle states |
| `TraceStatus` | unset, ok, error | Span completion status |
| `SLOStatus` | met, at_risk, breached | SLO compliance states |
| `AnomalyType` | spike, drop, trend, seasonality, outlier | Anomaly classifications |

### 2. Dataclasses

#### MetricPoint
```python
@dataclass
class MetricPoint:
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str]
    unit: str
```

#### MetricSeries
```python
@dataclass
class MetricSeries:
    name: str
    metric_type: MetricType
    points: List[MetricPoint]
    labels: Dict[str, str]

    def add_point(value: float) -> None
    def get_latest() -> Optional[MetricPoint]
    def get_average(window_seconds: int) -> Optional[float]
```

#### SpanContext
```python
@dataclass
class SpanContext:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    baggage: Dict[str, str]
    sampled: bool
```

#### Span
```python
@dataclass
class Span:
    context: SpanContext
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime]
    status: TraceStatus
    tags: Dict[str, Any]
    logs: List[Dict[str, Any]]
    duration_ms: float

    def finish(status: TraceStatus) -> None
    def log(message: str, **kwargs) -> None
```

#### HealthCheck
```python
@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float
    details: Dict[str, Any]
```

#### Alert
```python
@dataclass
class Alert:
    alert_id: str
    name: str
    severity: AlertSeverity
    state: AlertState
    message: str
    source: str
    labels: Dict[str, str]
    annotations: Dict[str, str]
    started_at: datetime
    resolved_at: Optional[datetime]
    firing_since: Optional[datetime]
    value: float
```

#### SLODefinition
```python
@dataclass
class SLODefinition:
    slo_id: str
    name: str
    target: float  # e.g., 99.9
    window_days: int
    metric_name: str
    description: str
    labels: Dict[str, str]
```

#### SLOResult
```python
@dataclass
class SLOResult:
    slo: SLODefinition
    current_value: float
    error_budget_remaining: float
    burn_rate: float
    status: str  # met, at_risk, breached
    timestamp: datetime
```

#### Anomaly
```python
@dataclass
class Anomaly:
    anomaly_id: str
    anomaly_type: AnomalyType
    metric_name: str
    expected_value: float
    actual_value: float
    deviation: float
    confidence: float
    timestamp: datetime
    details: Dict[str, Any]
```

#### ObservabilityConfig
```python
@dataclass
class ObservabilityConfig:
    metrics_retention_hours: int = 24
    trace_sample_rate: float = 0.1
    health_check_interval_seconds: int = 30
    alert_evaluation_interval_seconds: int = 60
    anomaly_detection_enabled: bool = True
    slo_tracking_enabled: bool = True
```

### 3. Core Classes

#### MetricsCollector
```python
class MetricsCollector:
    def increment(name: str, value: float = 1.0, labels: Dict = None) -> None
    def gauge(name: str, value: float, labels: Dict = None) -> None
    def histogram(name: str, value: float, labels: Dict = None) -> None
    def timer(name: str, duration_ms: float, labels: Dict = None) -> None
    def get_counter(name: str, labels: Dict = None) -> float
    def get_gauge(name: str, labels: Dict = None) -> Optional[float]
    def get_histogram_stats(name: str, labels: Dict = None) -> Dict[str, float]
    def get_metric(name: str, labels: Dict = None) -> Optional[MetricSeries]
    def list_metrics() -> List[str]
    def export() -> List[MetricPoint]
```

#### TraceManager
```python
class TraceManager:
    def start_span(operation_name: str, service_name: str,
                   parent_context: SpanContext = None, tags: Dict = None) -> Span
    def finish_span(span: Span, status: TraceStatus = TraceStatus.OK) -> None
    def get_trace(trace_id: str) -> List[Span]
    def get_active_spans() -> List[Span]
    def inject_context(span: Span) -> Dict[str, str]
    def extract_context(headers: Dict[str, str]) -> Optional[SpanContext]
```

#### HealthMonitor
```python
class HealthMonitor:
    def register_check(name: str, check_fn: Callable) -> None
    def unregister_check(name: str) -> bool
    def run_check(name: str) -> Optional[HealthCheck]
    def run_all_checks() -> Dict[str, HealthCheck]
    def get_overall_status() -> HealthStatus
    def get_check_result(name: str) -> Optional[HealthCheck]
    def list_checks() -> List[str]
```

#### AlertManager
```python
class AlertManager:
    def register_rule(name: str, rule_fn: Callable) -> None
    def unregister_rule(name: str) -> bool
    def add_handler(handler: Callable[[Alert], None]) -> None
    def add_suppression_rule(rule: Callable[[Alert], bool]) -> None
    def fire_alert(alert: Alert) -> None
    def resolve_alert(alert_id: str) -> Optional[Alert]
    def get_alert(alert_id: str) -> Optional[Alert]
    def get_firing_alerts() -> List[Alert]
    def get_alerts_by_severity(severity: AlertSeverity) -> List[Alert]
    def evaluate_rules(context: Dict[str, Any]) -> List[Alert]
```

#### SLOTracker
```python
class SLOTracker:
    def define_slo(slo: SLODefinition) -> None
    def record_event(slo_id: str, is_good: bool) -> None
    def get_slo_status(slo_id: str) -> Optional[SLOResult]
    def list_slos() -> List[SLODefinition]
```

#### AnomalyDetector
```python
class AnomalyDetector:
    def calculate_baseline(metric_name: str, labels: Dict = None) -> Optional[Dict]
    def detect(metric_name: str, value: float, labels: Dict = None) -> Optional[Anomaly]
    def get_detected_anomalies(limit: int = 100) -> List[Anomaly]
```

#### ObservabilityHub
```python
class ObservabilityHub:
    @property metrics: MetricsCollector
    @property traces: TraceManager
    @property health: HealthMonitor
    @property alerts: AlertManager
    @property slo_tracker: SLOTracker
    @property anomaly_detector: AnomalyDetector

    def start() -> None
    def stop() -> None
    def get_dashboard_data() -> Dict[str, Any]
```

### 4. Vision Provider Integration

#### ObservableVisionProvider
```python
class ObservableVisionProvider(VisionProvider):
    def __init__(base_provider: VisionProvider, hub: ObservabilityHub)

    @property provider_name: str  # "observable_{base_provider_name}"

    async def analyze_image(image_data: bytes, include_description: bool) -> VisionDescription
```

Auto-records:
- Request counts (`vision_requests_total`)
- Success/error counts (`vision_requests_success`, `vision_requests_error`)
- Duration metrics (`vision_request_duration_ms`)
- Confidence gauges (`vision_confidence`)
- Distributed traces for each request
- SLO events
- Anomaly detection on latency
- Alerts on errors

### 5. Factory Functions

```python
def create_observability_hub(
    metrics_retention_hours: int = 24,
    trace_sample_rate: float = 0.1,
    **kwargs
) -> ObservabilityHub

def create_metrics_collector(config: ObservabilityConfig = None) -> MetricsCollector
def create_trace_manager(config: ObservabilityConfig = None) -> TraceManager
def create_health_monitor(config: ObservabilityConfig = None) -> HealthMonitor
def create_alert_manager(config: ObservabilityConfig = None) -> AlertManager

def create_slo_definition(
    name: str,
    target: float,
    window_days: int = 30,
    metric_name: str = "",
    **kwargs
) -> SLODefinition

def create_observable_provider(
    base_provider: VisionProvider,
    hub: ObservabilityHub = None
) -> ObservableVisionProvider
```

## Usage Examples

### Basic Metrics Collection
```python
from src.core.vision import create_metrics_collector

metrics = create_metrics_collector()

# Counter
metrics.increment("api_requests", labels={"endpoint": "/analyze"})

# Gauge
metrics.gauge("active_connections", 42)

# Histogram
metrics.histogram("response_time_ms", 125.5)

# Get statistics
stats = metrics.get_histogram_stats("response_time_ms")
print(f"P99: {stats['p99']}ms")
```

### Distributed Tracing
```python
from src.core.vision import create_trace_manager, TraceStatus

traces = create_trace_manager()

# Start a trace
span = traces.start_span("process_image", "vision-service")
span.log("Starting image analysis")

try:
    # Do work
    result = process_image(data)
    traces.finish_span(span, TraceStatus.OK)
except Exception as e:
    span.log("error", error=str(e))
    traces.finish_span(span, TraceStatus.ERROR)
    raise
```

### Health Monitoring
```python
from src.core.vision import create_health_monitor, HealthCheck, HealthStatus

health = create_health_monitor()

def check_database():
    # Check DB connection
    return HealthCheck(
        name="database",
        status=HealthStatus.HEALTHY,
        message="Connected"
    )

health.register_check("database", check_database)
results = health.run_all_checks()
overall = health.get_overall_status()
```

### Alert Management
```python
from src.core.vision import create_alert_manager, Alert, AlertSeverity

alerts = create_alert_manager()

# Define alert rule
def high_latency_rule(ctx):
    if ctx.get("latency_ms", 0) > 1000:
        return Alert(
            alert_id="high-latency",
            name="HighLatency",
            severity=AlertSeverity.WARNING,
            value=ctx["latency_ms"]
        )
    return None

alerts.register_rule("latency", high_latency_rule)

# Add handler
alerts.add_handler(lambda a: send_slack_notification(a))

# Evaluate
fired = alerts.evaluate_rules({"latency_ms": 1500})
```

### SLO Tracking
```python
from src.core.vision import create_observability_hub, create_slo_definition

hub = create_observability_hub()

# Define SLO
slo = create_slo_definition(
    name="API Availability",
    target=99.9,
    window_days=30
)
hub.slo_tracker.define_slo(slo)

# Record events
hub.slo_tracker.record_event(slo.slo_id, is_good=True)
hub.slo_tracker.record_event(slo.slo_id, is_good=False)

# Check status
status = hub.slo_tracker.get_slo_status(slo.slo_id)
print(f"Current: {status.current_value}%, Budget: {status.error_budget_remaining}%")
```

### Full Observable Provider
```python
from src.core.vision import create_observable_provider, create_observability_hub

hub = create_observability_hub(trace_sample_rate=0.5)

# Wrap any provider
observable = create_observable_provider(my_provider, hub)

# All calls automatically tracked
result = await observable.analyze_image(image_data)

# Get dashboard data
dashboard = hub.get_dashboard_data()
```

## Test Coverage

| Test Class | Tests | Coverage |
|------------|-------|----------|
| TestObservabilityEnums | 7 | All enum values |
| TestObservabilityDataclasses | 12 | All dataclass creation and methods |
| TestMetricsCollector | 8 | All metric types and operations |
| TestTraceManager | 8 | Span lifecycle and context propagation |
| TestHealthMonitor | 9 | Check registration and status aggregation |
| TestAlertManager | 10 | Rules, handlers, suppression, resolution |
| TestSLOTracker | 6 | SLO definition, events, status calculation |
| TestAnomalyDetector | 6 | Baseline and anomaly detection |
| TestObservabilityHub | 4 | Hub integration |
| TestObservableVisionProvider | 5 | Provider integration |
| TestFactoryFunctions | 8 | All factory functions |
| TestObservabilityIntegration | 4 | End-to-end scenarios |

**Total: 85 tests**

## Integration with Existing Modules

Phase 21 complements existing observability modules:

| Existing Module | Phase 21 Enhancement |
|----------------|---------------------|
| `metrics_exporter.py` | ObservabilityHub provides unified metrics interface |
| `metrics_dashboard.py` | `get_dashboard_data()` method for dashboard integration |
| `tracing.py` | TraceManager adds context propagation |
| `distributed_tracing.py` | Compatible span context format |
| `health.py` | HealthMonitor with check aggregation |
| `alert_manager.py` | AlertManager with rules and suppression |

## Performance Considerations

- **Thread Safety**: All components use `threading.RLock()` for concurrent access
- **Trace Sampling**: Configurable `trace_sample_rate` (default 0.1 = 10%)
- **Metrics Retention**: Configurable retention hours
- **Lazy Initialization**: Components created on-demand
- **Efficient Key Generation**: Labels serialized to string keys

## Dependencies

- Standard library only (no external dependencies)
- Uses `threading`, `uuid`, `statistics`, `asyncio`
- Integrates with `VisionProvider` base class

## File Structure

```
src/core/vision/
├── observability_hub.py      # Phase 21 implementation
├── __init__.py               # Updated with Phase 21 exports
└── ...

tests/unit/
├── test_vision_phase21.py    # 85 comprehensive tests
└── ...
```
