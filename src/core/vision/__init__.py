# flake8: noqa: F401,F811,E501
"""Vision module for CAD ML Platform.

Provides vision-based analysis of engineering drawings with OCR integration.

Supported providers:
- stub: Testing stub (no API key required)
- deepseek: DeepSeek VL API
- openai: OpenAI GPT-4o/GPT-4-Vision
- anthropic: Claude 3 family (Opus, Sonnet, Haiku)

The experimental/ subpackage and the observability/enterprise scaffolding
(metrics, alerting, APM, SLA, chaos, multi-region, …) were removed in Phase 0
A2b — they were imported only by per-phase tests, never by a live route, and
are NOT part of the public API. See
docs/development/PHASE0_A2B_VISION_PRUNE_DESIGN_20260709.md.
"""


# Phase 14: Advanced Security & Privacy (merged with Phase 19)

from .base import (
    CadFeatureStats,
    OcrResult,
    VisionAnalyzeRequest,
    VisionAnalyzeResponse,
    VisionDescription,
    VisionError,
    VisionInputError,
    VisionProvider,
    VisionProviderError,
)

from .factory import create_vision_provider, get_available_providers, get_vision_provider

from .manager import VisionManager

# Phase 13: Advanced Distributed Systems & Messaging

# Phase 17: Advanced Observability & Monitoring

# Phase 12: Advanced Analytics & ML Integration

# Phase 21: Advanced Observability & Telemetry - Observability Hub

# Phase 16: Advanced Integration & Extensibility

# Phase 15: Advanced Analytics & Intelligence
from .providers import (  # Stub provider; DeepSeek provider; OpenAI provider; Anthropic provider
    AnthropicVisionProvider,
    DeepSeekStubProvider,
    DeepSeekVisionProvider,
    OpenAIVisionProvider,
    create_anthropic_provider,
    create_deepseek_provider,
    create_openai_provider,
    create_stub_provider,
)
from .resilience import (
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    ProviderMetrics,
    ResilientVisionProvider,
    RetryConfig,
    create_resilient_provider,
)

# Phase 22: Advanced Security & Governance - Security Governance Hub


# Phase 20: Advanced Integration & Extensibility - Webhook Handler

__all__ = [
    'VisionAnalyzeRequest',
    'VisionAnalyzeResponse',
    'VisionDescription',
    'OcrResult',
    'CadFeatureStats',
    'VisionProvider',
    'VisionManager',
    'DeepSeekStubProvider',
    'create_stub_provider',
    'DeepSeekVisionProvider',
    'create_deepseek_provider',
    'OpenAIVisionProvider',
    'create_openai_provider',
    'AnthropicVisionProvider',
    'create_anthropic_provider',
    'create_vision_provider',
    'get_vision_provider',
    'get_available_providers',
    'VisionError',
    'VisionProviderError',
    'VisionInputError',
    'ResilientVisionProvider',
    'RetryConfig',
    'CircuitBreakerConfig',
    'CircuitState',
    'CircuitOpenError',
    'ProviderMetrics',
    'create_resilient_provider',
]
