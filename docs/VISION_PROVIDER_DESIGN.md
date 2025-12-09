# Vision Provider System Design

## Overview

Multi-provider vision analysis system for engineering drawing interpretation, supporting
DeepSeek, OpenAI GPT-4o, and Anthropic Claude with automatic provider detection,
graceful fallback mechanisms, and production-grade resilience patterns.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API Layer                                       │
│  POST /analyze   GET /health   GET /providers   GET /metrics                │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Advanced Features                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │   Cache     │ │Rate Limiter │ │   Batch     │ │ Provider Comparison │   │
│  │   (LRU+TTL) │ │(Token Bucket)│ │ Processor   │ │  (Multi-Strategy)   │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Resilience Layer                                     │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │  Retry with         │  │  Circuit Breaker    │  │  Provider Metrics   │ │
│  │  Exponential Backoff│  │  CLOSED→OPEN→HALF   │  │  Tracking           │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VisionManager                                       │
│  - Orchestrates Vision + OCR analysis                                       │
│  - Singleton pattern with provider switching                                 │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Provider Factory                                      │
│  create_vision_provider(provider_type, fallback_to_stub)                    │
│                                                                              │
│  Priority: deepseek → openai → anthropic → stub                             │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────┐
          │                               │                           │
          ▼                               ▼                           ▼
┌───────────────────┐          ┌───────────────────┐        ┌───────────────────┐
│     DeepSeek      │          │      OpenAI       │        │    Anthropic      │
│     Provider      │          │     Provider      │        │     Provider      │
│                   │          │                   │        │                   │
│  deepseek-chat    │          │    gpt-4o         │        │  claude-sonnet    │
│  VL2 Model        │          │   GPT-4-turbo     │        │   Claude 3        │
└───────────────────┘          └───────────────────┘        └───────────────────┘
          │                               │                           │
          └───────────────────────────────┴───────────────────────────┘
                                          │
                                          ▼
                                ┌───────────────────┐
                                │       Stub        │
                                │     Provider      │
                                │    (Fallback)     │
                                └───────────────────┘
```

## Provider Implementations

### 1. DeepSeekVisionProvider

**File**: `src/core/vision/providers/deepseek.py`

- **API**: DeepSeek Platform API (https://api.deepseek.com)
- **Model**: `deepseek-chat` (VL2 multimodal support)
- **Authentication**: Bearer token via `DEEPSEEK_API_KEY`
- **Features**:
  - Engineering drawing analysis prompt
  - JSON response with fallback parsing
  - Image type detection (PNG, JPEG, GIF, WebP)
  - Configurable timeout and max tokens

### 2. OpenAIVisionProvider

**File**: `src/core/vision/providers/openai.py`

- **API**: OpenAI Chat Completions (https://api.openai.com/v1)
- **Model**: `gpt-4o` (default), `gpt-4-turbo` supported
- **Authentication**: Bearer token via `OPENAI_API_KEY`
- **Features**:
  - `response_format: {"type": "json_object"}` for reliable JSON
  - Detail level control (`low`, `high`, `auto`)
  - Azure OpenAI compatible via base_url override
  - System prompt for engineering drawings

### 3. AnthropicVisionProvider

**File**: `src/core/vision/providers/anthropic.py`

- **API**: Anthropic Messages API (https://api.anthropic.com)
- **Model**: `claude-sonnet-4-20250514` (default)
- **Authentication**: `x-api-key` header via `ANTHROPIC_API_KEY`
- **Features**:
  - Claude's base64 image source format
  - Multiple JSON extraction strategies
  - Longer timeout (90s) for complex analysis
  - Support for Opus, Sonnet, Haiku variants

### 4. DeepSeekStubProvider

**File**: `src/core/vision/providers/deepseek_stub.py`

- **Purpose**: Testing and development without API keys
- **Features**:
  - Simulated latency option
  - Fixed response for consistent testing
  - No external dependencies

## Factory Pattern

### Provider Selection

```python
def create_vision_provider(
    provider_type: Optional[str] = None,  # deepseek, openai, anthropic, stub, auto
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    fallback_to_stub: bool = True,
    **kwargs,
) -> VisionProvider
```

### Auto-Detection Priority

1. Check `DEEPSEEK_API_KEY` → DeepSeekVisionProvider
2. Check `OPENAI_API_KEY` → OpenAIVisionProvider
3. Check `ANTHROPIC_API_KEY` → AnthropicVisionProvider
4. Fallback → DeepSeekStubProvider

### Fallback Mechanism

When `fallback_to_stub=True` (default):
- Provider initialization failure → stub provider
- Missing API key → stub provider
- Unknown provider type → raises error

---

## Advanced Features

### 1. Response Caching

**File**: `src/core/vision/cache.py`

LRU cache with TTL for vision responses to reduce API calls and latency.

```python
from src.core.vision import create_cached_provider

# Create cached provider
provider = create_vision_provider("openai")
cached = create_cached_provider(provider, max_size=100, ttl_seconds=3600)

# First call hits API
result1 = await cached.analyze_image(image_bytes)

# Second call returns cached result
result2 = await cached.analyze_image(image_bytes)  # Cache hit!

# Check cache statistics
stats = cached.cache_stats
print(f"Hits: {stats.hits}, Misses: {stats.misses}")
print(f"Hit rate: {stats.hit_rate:.1%}")
```

**Cache Features**:
- **LRU Eviction**: Removes least recently used entries when full
- **TTL Expiration**: Auto-expire stale entries
- **Content-Based Keys**: SHA-256 hash of image + provider + settings
- **Async-Safe**: Thread-safe operations with asyncio locks

**CacheStats**:
```python
@dataclass
class CacheStats:
    hits: int          # Cache hit count
    misses: int        # Cache miss count
    size: int          # Current cache size
    evictions: int     # Entries evicted

    @property
    def hit_rate(self) -> float  # hits / total queries
```

### 2. Rate Limiting

**File**: `src/core/vision/rate_limiter.py`

Token bucket rate limiter to respect API limits and prevent throttling.

```python
from src.core.vision import create_rate_limited_provider, RateLimitConfig

# Create rate-limited provider
provider = create_vision_provider("openai")
limited = create_rate_limited_provider(
    provider,
    requests_per_minute=60,
    burst_size=10,
)

# Use normally - rate limiting is automatic
result = await limited.analyze_image(image_bytes)

# Check available capacity
print(f"Available requests: {limited.available_requests}")

# Or get full stats
stats = limited.rate_limit_stats
print(f"Total: {stats.total_requests}, Rejected: {stats.rejected_requests}")
```

**Default Rate Limits per Provider**:
| Provider | RPM | Burst Size |
|----------|-----|------------|
| OpenAI | 60 | 10 |
| Anthropic | 60 | 10 |
| DeepSeek | 120 | 20 |
| Stub | 1000 | 100 |

**Rate Limit Options**:
- **Wait Mode**: Block until tokens available (default)
- **Reject Mode**: Raise `RateLimitError` immediately
- **Custom Config**: Override defaults per use case

```python
# Reject immediately if rate limited
try:
    result = await limited.analyze_image(
        image_bytes,
        wait_for_rate_limit=False,
    )
except RateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after}s")
```

### 3. Batch Processing

**File**: `src/core/vision/batch.py`

Concurrent batch processing with progress tracking and error isolation.

```python
from src.core.vision import process_images_batch, BatchProcessor

# Simple batch processing
provider = create_vision_provider("openai")
images = [img1, img2, img3, img4, img5]

result = await process_images_batch(
    provider,
    images,
    max_concurrency=3,
)

print(f"Completed: {result.completed}/{result.total}")
print(f"Success rate: {result.success_rate:.1%}")
print(f"Time: {result.total_time_ms:.0f}ms")

# Access individual results
for i, desc in enumerate(result.results):
    if desc:
        print(f"Image {i}: {desc.summary}")
    else:
        print(f"Image {i}: Error - {result.errors.get(i)}")
```

**With Progress Callback**:
```python
def on_progress(progress):
    print(f"Progress: {progress.progress_percent:.0f}%")
    print(f"ETA: {progress.estimated_remaining_seconds:.0f}s")

result = await process_images_batch(
    provider,
    images,
    max_concurrency=5,
    progress_callback=on_progress,
)
```

**BatchResult**:
```python
@dataclass
class BatchResult:
    total: int                              # Total images
    completed: int                          # Successful analyses
    failed: int                             # Failed analyses
    results: List[Optional[VisionDescription]]  # Results by index
    errors: Dict[int, str]                  # Errors by index
    total_time_ms: float                    # Total processing time

    @property
    def success_rate(self) -> float         # completed / total
```

### 4. Provider Comparison

**File**: `src/core/vision/comparison.py`

Compare multiple providers on the same image and select the best result.

```python
from src.core.vision import (
    compare_providers,
    ProviderComparator,
    SelectionStrategy,
    create_vision_provider,
)

# Create multiple providers
providers = [
    create_vision_provider("openai"),
    create_vision_provider("anthropic"),
    create_vision_provider("deepseek"),
]

# Simple comparison
result = await compare_providers(image_bytes, providers)

print(f"Best provider: {result.selected_provider}")
print(f"Confidence: {result.selected_result.confidence}")
print(f"All scores: {result.confidence_scores}")
```

**Selection Strategies**:

| Strategy | Description |
|----------|-------------|
| `HIGHEST_CONFIDENCE` | Select result with highest confidence score (default) |
| `FIRST_SUCCESS` | Return first successful result |
| `MAJORITY_VOTE` | Select most common result pattern |
| `WEIGHTED_AVERAGE` | Combine results weighted by confidence |

```python
# Use specific strategy
comparator = ProviderComparator(
    providers,
    selection_strategy=SelectionStrategy.WEIGHTED_AVERAGE,
)
result = await comparator.compare(image_bytes)
```

**ComparisonResult**:
```python
@dataclass
class ComparisonResult:
    provider_results: Dict[str, ProviderResult]  # All provider results
    selected_result: Optional[VisionDescription]  # Best result
    selected_provider: Optional[str]              # Best provider name
    aggregated_summary: Optional[str]             # Combined summaries
    total_time_ms: float                          # Total comparison time
    strategy_used: SelectionStrategy              # Strategy applied

    @property
    def success_count(self) -> int                # Successful providers

    @property
    def providers_compared(self) -> int           # Total providers

    @property
    def confidence_scores(self) -> Dict[str, float]  # Per-provider confidence
```

---

## Resilience Patterns

### Retry with Exponential Backoff

```python
from src.core.vision import create_resilient_provider, RetryConfig

# Create provider with custom retry config
provider = create_vision_provider("openai")
resilient = create_resilient_provider(
    provider,
    max_retries=3,
    circuit_failure_threshold=5,
    circuit_timeout=60.0,
)

result = await resilient.analyze_image(image_bytes)
```

**RetryConfig Options**:
- `max_retries`: Maximum retry attempts (default: 3)
- `base_delay`: Initial delay in seconds (default: 1.0)
- `max_delay`: Maximum delay cap (default: 30.0)
- `exponential_base`: Backoff multiplier (default: 2.0)

### Circuit Breaker Pattern

Protects against cascading failures:

```
┌─────────────────────────────────────────────────────────────┐
│                    Circuit Breaker States                   │
├─────────────┬───────────────┬───────────────────────────────┤
│   CLOSED    │   OPEN        │   HALF_OPEN                   │
│   (Normal)  │   (Failing)   │   (Testing)                   │
├─────────────┼───────────────┼───────────────────────────────┤
│ Requests    │ Requests      │ Limited test                  │
│ pass thru   │ rejected      │ requests allowed              │
└─────────────┴───────────────┴───────────────────────────────┘
         │                         │
         │  failure_threshold      │  success_threshold
         │  exceeded               │  reached
         ▼                         ▼
      [OPEN] ──── timeout ────▶ [HALF_OPEN] ──── success ────▶ [CLOSED]
         ▲                         │
         └───── test failure ──────┘
```

**CircuitBreakerConfig Options**:
- `failure_threshold`: Failures before opening (default: 5)
- `success_threshold`: Successes to close from half-open (default: 2)
- `timeout`: Seconds before testing half-open (default: 60.0)
- `half_open_max_calls`: Max concurrent calls in half-open (default: 1)

### Provider Metrics

Track performance with built-in metrics:

```python
resilient = create_resilient_provider(provider)

# After some requests...
metrics = resilient.metrics
print(f"Success rate: {metrics.success_rate:.2%}")
print(f"Avg latency: {metrics.average_latency_ms:.0f}ms")
print(f"Circuit state: {resilient.circuit_state.value}")
```

**Available Metrics**:
- `total_requests`: Total API calls
- `successful_requests`: Successful calls
- `failed_requests`: Failed calls
- `success_rate`: Success ratio (0.0-1.0)
- `average_latency_ms`: Mean response time
- `last_error`: Most recent error message
- `circuit_opens`: Times circuit opened

---

## API Endpoints

### POST /api/v1/vision/analyze

Analyze engineering drawing with optional provider override.

**Request**:
```json
{
    "image_base64": "iVBORw0KGgoAAAANS...",
    "include_description": true,
    "include_ocr": true,
    "ocr_provider": "auto"
}
```

**Query Parameters**:
- `provider`: Override provider (deepseek, openai, anthropic, stub)

**Response**:
```json
{
    "success": true,
    "description": {
        "summary": "Mechanical part with cylindrical features",
        "details": ["Main diameter: 20mm", "Thread: M10x1.5"],
        "confidence": 0.92
    },
    "ocr": null,
    "provider": "openai",
    "processing_time_ms": 1234.5
}
```

### GET /api/v1/vision/health

Service health check.

**Response**:
```json
{
    "status": "healthy",
    "provider": "openai",
    "ocr_enabled": true
}
```

### GET /api/v1/vision/providers

List available providers and their status.

**Response**:
```json
{
    "current_provider": "openai",
    "providers": {
        "stub": {"available": true, "requires_key": false},
        "deepseek": {"available": true, "requires_key": true, "key_set": false},
        "openai": {"available": true, "requires_key": true, "key_set": true},
        "anthropic": {"available": true, "requires_key": true, "key_set": false}
    }
}
```

### GET /api/v1/vision/metrics

Performance metrics endpoint.

**Response**:
```json
{
    "provider": "openai",
    "resilient": true,
    "metrics": {
        "total_requests": 100,
        "successful_requests": 95,
        "failed_requests": 5,
        "success_rate": 0.95,
        "average_latency_ms": 1234.5,
        "last_error": "Timeout after 60s",
        "circuit_state": "closed",
        "circuit_opens": 1
    }
}
```

---

## Data Models

### VisionDescription

```python
class VisionDescription(BaseModel):
    summary: str           # Brief description of the drawing
    details: List[str]     # Specific observations
    confidence: float      # 0.0 - 1.0 confidence score
```

### VisionAnalyzeRequest

```python
class VisionAnalyzeRequest(BaseModel):
    image_base64: str
    include_description: bool = True
    include_ocr: bool = True
    ocr_provider: Optional[str] = None
```

### VisionAnalyzeResponse

```python
class VisionAnalyzeResponse(BaseModel):
    success: bool
    description: Optional[VisionDescription]
    ocr: Optional[OcrResult]
    provider: str
    processing_time_ms: float
    error: Optional[str]
    code: Optional[str]
```

---

## Configuration

### Environment Variables

```bash
# Provider selection (auto, deepseek, openai, anthropic, stub)
VISION_PROVIDER=auto

# API Keys
DEEPSEEK_API_KEY=sk-...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Provider-Specific Config

```python
# DeepSeek
DeepSeekVisionProvider(
    api_key=None,           # or DEEPSEEK_API_KEY
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    timeout_seconds=60.0,
    max_tokens=2048,
)

# OpenAI
OpenAIVisionProvider(
    api_key=None,           # or OPENAI_API_KEY
    base_url="https://api.openai.com/v1",
    model="gpt-4o",
    timeout_seconds=60.0,
    max_tokens=2048,
    detail="high",          # low, high, auto
)

# Anthropic
AnthropicVisionProvider(
    api_key=None,           # or ANTHROPIC_API_KEY
    base_url="https://api.anthropic.com",
    model="claude-sonnet-4-20250514",
    timeout_seconds=90.0,
    max_tokens=2048,
)
```

---

## Error Handling

### Exception Hierarchy

```python
class VisionError(Exception):
    """Base exception for vision module."""

class VisionProviderError(VisionError):
    """Provider-specific errors."""
    provider: str
    message: str

class VisionInputError(VisionError):
    """Invalid input errors."""

class RateLimitError(VisionProviderError):
    """Rate limit exceeded."""
    retry_after: float

class CircuitOpenError(VisionProviderError):
    """Circuit breaker is open."""
```

### Error Codes

| Code | Description |
|------|-------------|
| `INPUT_ERROR` | Invalid input (empty image, wrong format) |
| `EXTERNAL_SERVICE_ERROR` | Provider API error |
| `INTERNAL_ERROR` | Unexpected internal error |
| `RATE_LIMIT_ERROR` | Rate limit exceeded |
| `CIRCUIT_OPEN_ERROR` | Circuit breaker tripped |

### Error Recovery

1. API timeout → Raise VisionProviderError with timeout info
2. Invalid JSON response → Fallback to raw text summary
3. Provider unavailable → Fall back to stub (if enabled)
4. Empty response → Raise VisionProviderError
5. Rate limited → Wait or raise RateLimitError
6. Circuit open → Raise CircuitOpenError

---

## Testing

### Test Coverage

- **70 unit tests** covering:
  - Provider initialization with/without API keys
  - Image type detection (PNG, JPEG, GIF, WebP)
  - OCR-only mode responses
  - Factory pattern with all provider types
  - Auto-detection priority
  - Fallback mechanism
  - Error handling
  - Retry logic with exponential backoff
  - Circuit breaker state transitions
  - Metrics collection and calculation
  - Cache hit/miss/expiration/eviction
  - Rate limiter token bucket algorithm
  - Batch processing with concurrency control
  - Provider comparison with all strategies

- **Integration tests** covering:
  - Full API request/response cycle
  - Provider switching via query parameter
  - Health check endpoint
  - Providers listing endpoint

### Running Tests

```bash
# Unit tests
pytest tests/unit/test_vision_providers.py -v
pytest tests/unit/test_vision_resilience.py -v
pytest tests/unit/test_vision_advanced.py -v

# Integration tests
pytest tests/integration/test_vision_api_integration.py -v

# All vision tests
pytest tests/ -k "vision" -v
```

---

## File Structure

```
src/core/vision/
├── __init__.py              # Public exports (all components)
├── base.py                  # VisionProvider ABC, VisionDescription
├── factory.py               # create_vision_provider, get_available_providers
├── manager.py               # VisionManager orchestration
├── resilience.py            # Retry, circuit breaker, metrics
├── cache.py                 # LRU cache with TTL
├── rate_limiter.py          # Token bucket rate limiter
├── batch.py                 # Concurrent batch processing
├── comparison.py            # Multi-provider comparison
└── providers/
    ├── __init__.py          # Provider exports
    ├── anthropic.py         # AnthropicVisionProvider
    ├── deepseek.py          # DeepSeekVisionProvider
    ├── deepseek_stub.py     # DeepSeekStubProvider (testing)
    └── openai.py            # OpenAIVisionProvider

src/api/v1/
└── vision.py                # FastAPI router with endpoints

tests/unit/
├── test_vision_providers.py  # Provider unit tests (32 tests)
├── test_vision_resilience.py # Resilience pattern tests (19 tests)
└── test_vision_advanced.py   # Advanced feature tests (19 tests)

tests/integration/
└── test_vision_api_integration.py # API integration tests
```

---

## Usage Examples

### Basic Usage

```python
from src.core.vision import create_vision_provider

# Auto-detect provider based on available API keys
provider = create_vision_provider()
result = await provider.analyze_image(image_bytes)
print(f"Summary: {result.summary}")
print(f"Confidence: {result.confidence}")
```

### Production Stack

```python
from src.core.vision import (
    create_vision_provider,
    create_resilient_provider,
    create_cached_provider,
    create_rate_limited_provider,
)

# Build production-ready provider stack
base = create_vision_provider("openai")

# Add resilience (retry + circuit breaker)
resilient = create_resilient_provider(
    base,
    max_retries=3,
    circuit_failure_threshold=5,
)

# Add caching
cached = create_cached_provider(resilient, max_size=100, ttl_seconds=3600)

# Add rate limiting
provider = create_rate_limited_provider(cached, requests_per_minute=60)

# Use with full protection
result = await provider.analyze_image(image_bytes)
```

### Batch Processing

```python
from src.core.vision import process_images_batch

images = load_images_from_directory("./drawings/")
result = await process_images_batch(
    provider,
    images,
    max_concurrency=5,
    progress_callback=lambda p: print(f"{p.progress_percent:.0f}%"),
)
print(f"Processed {result.completed}/{result.total} images")
```

### Provider Comparison

```python
from src.core.vision import compare_providers, SelectionStrategy

providers = [
    create_vision_provider("openai"),
    create_vision_provider("anthropic"),
]

result = await compare_providers(
    image_bytes,
    providers,
    strategy=SelectionStrategy.HIGHEST_CONFIDENCE,
)
print(f"Best: {result.selected_provider} ({result.selected_result.confidence:.2f})")
```

### API Usage

```bash
# Analyze with auto-detected provider
curl -X POST "http://localhost:8000/api/v1/vision/analyze" \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "...", "include_description": true}'

# Force specific provider
curl -X POST "http://localhost:8000/api/v1/vision/analyze?provider=anthropic" \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "..."}'

# Check available providers
curl "http://localhost:8000/api/v1/vision/providers"

# Get performance metrics
curl "http://localhost:8000/api/v1/vision/metrics"
```

---

## Extended Features

### 5. Streaming Support

**File**: `src/core/vision/streaming.py`

Real-time streaming of analysis results using Server-Sent Events (SSE).

```python
from src.core.vision import create_streaming_provider, StreamEventType

# Create streaming provider
provider = create_vision_provider("openai")
streaming = create_streaming_provider(provider)

# Stream analysis
async for event in streaming.analyze_image_stream(image_bytes):
    if event.event_type == StreamEventType.PROGRESS:
        print(f"Progress: {event.data['progress']:.0%}")
    elif event.event_type == StreamEventType.COMPLETE:
        print(f"Result: {event.data['result']['summary']}")
```

**Event Types**:
| Event | Description |
|-------|-------------|
| `START` | Analysis started |
| `PROGRESS` | Progress update |
| `PARTIAL` | Partial result |
| `COMPLETE` | Analysis complete with result |
| `ERROR` | Error occurred |
| `HEARTBEAT` | Keep-alive signal |

**SSE Integration (FastAPI)**:
```python
from fastapi.responses import StreamingResponse

@router.get("/analyze/stream")
async def stream_analysis(image_base64: str):
    image_data = base64.b64decode(image_base64)
    return StreamingResponse(
        generate_sse_stream(provider, image_data),
        media_type="text/event-stream",
    )
```

### 6. Custom Prompts

**File**: `src/core/vision/prompts.py`

Configurable prompt templates for different analysis types.

```python
from src.core.vision import get_prompts, register_custom_template, PromptType

# Use built-in template
system, user = get_prompts("engineering_drawing")

# Register custom template
template = register_custom_template(
    name="pcb_layout",
    system_prompt="You are a PCB design expert...",
    user_prompt="Analyze this PCB for {focus_area}",
    variables=["focus_area"],
)

# Use custom template
system, user = get_prompts("pcb_layout", focus_area="thermal issues")
```

**Built-in Templates**:
| Template | Use Case |
|----------|----------|
| `engineering_drawing` | Mechanical drawings (default) |
| `architectural` | Floor plans, elevations |
| `circuit_diagram` | Electrical schematics |
| `flowchart` | Process diagrams |
| `general` | Generic image analysis |

### 7. Cost Tracking

**File**: `src/core/vision/cost_tracker.py`

Track API costs with budget limits and alerts.

```python
from src.core.vision import (
    create_cost_tracked_provider,
    BudgetConfig,
)

# Create with budget limits
tracked = create_cost_tracked_provider(
    provider,
    budget_config=BudgetConfig(
        daily_limit_usd=10.0,
        monthly_limit_usd=100.0,
        alert_threshold_percent=80.0,
        hard_limit=True,  # Reject requests over limit
    ),
)

# Use normally
result = await tracked.analyze_image(image_bytes)

# Check costs
tracker = tracked.cost_tracker
print(f"Daily cost: ${tracker.get_daily_cost():.2f}")
print(f"Monthly cost: ${tracker.get_monthly_cost():.2f}")

# Get detailed summary
summary = tracker.get_usage_summary(provider="openai")
print(f"Total requests: {summary.total_requests}")
print(f"Success rate: {summary.success_rate:.0%}")
print(f"Avg cost per request: ${summary.cost_per_request:.4f}")
```

**Default Pricing** (approximate):
| Provider | Input/1K tokens | Output/1K tokens | Per image |
|----------|-----------------|------------------|-----------|
| OpenAI | $0.005 | $0.015 | $0.00765 |
| Anthropic | $0.003 | $0.015 | $0.0048 |
| DeepSeek | $0.00014 | $0.00028 | $0.001 |

### 8. Webhook Notifications

**File**: `src/core/vision/webhooks.py`

Async notifications for analysis events.

```python
from src.core.vision import (
    create_webhook_provider,
    WebhookConfig,
    WebhookEventType,
)

# Create webhook-enabled provider
webhook_provider = create_webhook_provider(
    provider,
    webhook_url="https://api.example.com/webhooks/vision",
    webhook_secret="my-secret-key",
)

# Start the delivery worker
await webhook_provider.webhook_manager.start()

# Analyze (webhooks sent automatically)
result = await webhook_provider.analyze_image(image_bytes)
```

**Webhook Events**:
| Event | Trigger |
|-------|---------|
| `analysis.started` | Analysis begins |
| `analysis.completed` | Analysis succeeds |
| `analysis.failed` | Analysis fails |
| `batch.started` | Batch processing begins |
| `batch.progress` | Batch progress update |
| `batch.completed` | Batch completes |
| `budget.alert` | Budget threshold reached |

**Webhook Payload Example**:
```json
{
    "event_id": "abc123",
    "event_type": "analysis.completed",
    "payload": {
        "request_id": "req-456",
        "provider": "openai",
        "elapsed_ms": 1234.5,
        "result": {
            "summary": "Engineering drawing analysis...",
            "confidence": 0.92
        }
    },
    "timestamp": "2024-01-15T10:30:00Z"
}
```

**Signature Verification**:
```python
# On your webhook receiver
from src.core.vision import WebhookManager

is_valid = WebhookManager.verify_signature(
    payload=request.body,
    signature=request.headers["X-Webhook-Signature"],
    secret="my-secret-key",
)
```

### 9. Result Persistence

**File**: `src/core/vision/persistence.py`

Store and retrieve analysis results with query capabilities.

```python
from src.core.vision import (
    create_persistent_provider,
    ResultPersistence,
    QueryFilter,
)

# Create persistent provider
persistent = create_persistent_provider(
    provider,
    auto_tag=["production", "batch-1"],
)

# Analyze and store
result, record = await persistent.analyze_image(
    image_bytes,
    tags=["mechanical", "urgent"],
)
print(f"Saved as: {record.record_id}")

# Query results
persistence = persistent.persistence
results = await persistence.query_results(
    provider="openai",
    min_confidence=0.8,
    limit=50,
)
for record in results.records:
    print(f"{record.record_id}: {record.result.summary}")
```

**Storage Backends**:
| Backend | Description |
|---------|-------------|
| `InMemoryStorage` | Testing and development (default) |
| `SQLiteStorage` | Local file-based storage |
| `PostgreSQLStorage` | Production database |
| `RedisStorage` | High-performance cache |

**AnalysisRecord Fields**:
```python
@dataclass
class AnalysisRecord:
    record_id: str           # Unique ID
    image_hash: str          # SHA-256 of image
    provider: str            # Provider used
    result: VisionDescription # Analysis result
    created_at: datetime     # Timestamp
    processing_time_ms: float # Duration
    cost_usd: float          # API cost
    tags: List[str]          # Organization tags
```

### 10. Analytics & Reporting

**File**: `src/core/vision/analytics.py`

Generate insights from analysis history.

```python
from src.core.vision import create_analytics, TimeGranularity

# Create analytics engine
analytics = create_analytics(persistence)

# Get provider statistics
stats = await analytics.get_provider_stats()
for name, stat in stats.items():
    print(f"{name}: {stat.success_rate:.0%} success, ${stat.avg_cost_per_request:.4f}/req")

# Get trends over time
trends = await analytics.get_trends(
    granularity=TimeGranularity.DAY,
    days=30,
)
print(f"Request trend: {trends.request_trend}")
print(f"Cost trend: ${sum(trends.cost_trend):.2f} total")

# Generate comprehensive report
report = await analytics.generate_report()
print(f"Total requests: {report.overall_stats.total_requests}")
print(f"Insights: {report.insights}")
```

**Report Contents**:
- Provider-level statistics (requests, success rate, costs)
- Time-series trends (hourly, daily, weekly, monthly)
- Top tags analysis
- Automated insights generation
- Confidence score distribution

**AnalyticsReport**:
```python
@dataclass
class AnalyticsReport:
    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    provider_stats: Dict[str, ProviderStats]
    overall_stats: ProviderStats
    trends: Optional[TrendData]
    top_tags: List[tuple[str, int]]
    insights: List[str]
```

### 11. Multi-Region Failover

**File**: `src/core/vision/failover.py`

Geographic failover with multiple strategies.

```python
from src.core.vision import (
    create_failover_provider,
    FailoverStrategy,
    FailoverConfig,
)

# Create failover provider
providers = [
    create_vision_provider("openai"),    # Primary
    create_vision_provider("anthropic"), # Secondary
    create_vision_provider("deepseek"),  # Tertiary
]

failover = create_failover_provider(
    providers,
    strategy=FailoverStrategy.PRIORITY,
    max_retries=3,
)

# Analyze with automatic failover
result = await failover.analyze_image(image_bytes)

# Check status
status = failover.failover_manager.get_status()
print(f"Healthy endpoints: {status['healthy_count']}/{status['total_count']}")
```

**Failover Strategies**:
| Strategy | Description |
|----------|-------------|
| `PRIORITY` | Use providers in priority order (default) |
| `ROUND_ROBIN` | Rotate through healthy providers |
| `LATENCY` | Choose lowest latency provider |
| `WEIGHTED` | Random selection weighted by provider weight |

**Health States**:
| State | Description |
|-------|-------------|
| `HEALTHY` | Normal operation |
| `DEGRADED` | Experiencing issues (1-2 failures) |
| `UNHEALTHY` | Not available (5+ consecutive failures) |
| `UNKNOWN` | Not yet checked |

**ProviderEndpoint Tracking**:
```python
@dataclass
class ProviderEndpoint:
    provider: VisionProvider
    priority: int              # Lower = higher priority
    weight: float              # For weighted selection
    region: str                # Geographic region
    health: ProviderHealth     # Current health state
    consecutive_failures: int  # Failure count
    avg_latency_ms: float      # Average response time
```

### 12. Health Monitoring

**File**: `src/core/vision/health.py`

Continuous health monitoring with alerting.

```python
from src.core.vision import (
    create_health_aware_provider,
    HealthMonitor,
    HealthCheckConfig,
    AlertSeverity,
)

# Create health monitor
monitor = HealthMonitor(config=HealthCheckConfig(
    interval_seconds=60.0,
    failure_threshold=3,
    latency_degraded_ms=5000.0,
))

# Register providers
monitor.register_provider(create_vision_provider("openai"))
monitor.register_provider(create_vision_provider("anthropic"))

# Add alert callback
def on_alert(alert):
    if alert.severity == AlertSeverity.CRITICAL:
        send_pager_duty_alert(alert)
    else:
        send_slack_notification(alert)

monitor.add_alert_callback(on_alert)

# Start monitoring
await monitor.start()

# Get dashboard
dashboard = monitor.get_dashboard()
print(f"Overall status: {dashboard.overall_status.value}")
print(f"Active alerts: {len(dashboard.active_alerts)}")
```

**Health Metrics**:
```python
@dataclass
class HealthMetrics:
    provider: str
    status: HealthStatus       # HEALTHY, DEGRADED, UNHEALTHY
    uptime_percentage: float   # Overall uptime
    avg_latency_ms: float      # Average response time
    p95_latency_ms: float      # 95th percentile latency
    p99_latency_ms: float      # 99th percentile latency
    success_rate: float        # Request success rate
    total_checks: int          # Total health checks
    consecutive_failures: int  # Current failure streak
```

**Alert Severities**:
| Severity | Trigger |
|----------|---------|
| `INFO` | Recovery from degraded/unhealthy |
| `WARNING` | Provider degraded |
| `ERROR` | Provider temporarily unhealthy |
| `CRITICAL` | Provider persistently unhealthy |

**Health-Aware Provider**:
```python
# Auto-update metrics on every request
aware = create_health_aware_provider(provider)
result = await aware.analyze_image(image_bytes)

# Check current health
health = aware.get_health()
print(f"Status: {health.status.value}")
print(f"Success rate: {health.success_rate:.0%}")
```

---

## Summary

The Vision Provider System provides a complete, production-ready solution for engineering drawing analysis:

| Feature | Description | Status |
|---------|-------------|--------|
| Multi-Provider | DeepSeek, OpenAI, Anthropic support | ✅ |
| Auto-Detection | Automatic provider selection by API key | ✅ |
| Fallback | Graceful fallback to stub provider | ✅ |
| Retry | Exponential backoff retry logic | ✅ |
| Circuit Breaker | Cascading failure protection | ✅ |
| Metrics | Performance tracking and reporting | ✅ |
| Caching | LRU cache with TTL | ✅ |
| Rate Limiting | Token bucket per provider | ✅ |
| Batch Processing | Concurrent image processing | ✅ |
| Provider Comparison | Multi-provider analysis | ✅ |
| Streaming | Real-time SSE streaming | ✅ |
| Custom Prompts | Configurable analysis prompts | ✅ |
| Cost Tracking | Usage monitoring with budgets | ✅ |
| Webhooks | Async event notifications | ✅ |
| Result Persistence | Database storage with queries | ✅ |
| Analytics | Reports and trend analysis | ✅ |
| Multi-Region Failover | Geographic redundancy | ✅ |
| Health Monitoring | Continuous health checks | ✅ |
| API Endpoints | REST API integration | ✅ |
| Unit Tests | 136 comprehensive tests | ✅ |
| Integration Tests | Full API test coverage | ✅ |

## File Structure (Complete)

```
src/core/vision/
├── __init__.py              # Public exports (all components)
├── base.py                  # VisionProvider ABC, VisionDescription
├── factory.py               # create_vision_provider, get_available_providers
├── manager.py               # VisionManager orchestration
├── resilience.py            # Retry, circuit breaker, metrics
├── cache.py                 # LRU cache with TTL
├── rate_limiter.py          # Token bucket rate limiter
├── batch.py                 # Concurrent batch processing
├── comparison.py            # Multi-provider comparison
├── streaming.py             # SSE streaming support
├── prompts.py               # Custom prompt templates
├── cost_tracker.py          # Usage and cost tracking
├── webhooks.py              # Webhook notifications
├── persistence.py           # Result storage and queries
├── analytics.py             # Analytics and reporting
├── failover.py              # Multi-region failover
├── health.py                # Health monitoring
└── providers/
    ├── __init__.py          # Provider exports
    ├── anthropic.py         # AnthropicVisionProvider
    ├── deepseek.py          # DeepSeekVisionProvider
    ├── deepseek_stub.py     # DeepSeekStubProvider (testing)
    └── openai.py            # OpenAIVisionProvider

tests/unit/
├── test_vision_providers.py  # Provider unit tests (32 tests)
├── test_vision_resilience.py # Resilience pattern tests (19 tests)
├── test_vision_advanced.py   # Advanced feature tests (19 tests)
├── test_vision_extended.py   # Extended feature tests (30 tests)
└── test_vision_persistence.py # Persistence/analytics tests (36 tests)
```

## Future Enhancements

1. **Azure OpenAI**: Explicit Azure configuration support
2. **PostgreSQL Storage**: Production database backend
3. **Prometheus Metrics**: Export to monitoring systems
4. **A/B Testing**: Provider comparison analytics
5. **Multi-Model Support**: Use different models per provider
