"""Vision module for CAD ML Platform.

Provides vision-based analysis of engineering drawings with OCR integration.

Supported providers:
- stub: Testing stub (no API key required)
- deepseek: DeepSeek VL API
- openai: OpenAI GPT-4o/GPT-4-Vision
- anthropic: Claude 3 family (Opus, Sonnet, Haiku)
- qwen_vl / glm4v / doubao: China-region vision providers
- vllm_vision: Self-hosted vLLM vision backend

Phase 0 slice A2b (this prune): removed ~90 unused enterprise-scaffold
submodules (ab_testing, access_control, plugin_system, saga_pattern, ...)
that were never imported by any live entry point. The surviving surface is
exactly the transitive-import closure of:
  - src/api/v1/vision.py  (the registered vision router)
  - src/core/providers/vision.py  (the core-provider bridge)
  - create_vision_provider(...) and its 7 concrete provider backends
"""

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


__all__ = [
    # Models
    "VisionAnalyzeRequest",
    "VisionAnalyzeResponse",
    "VisionDescription",
    "OcrResult",
    "CadFeatureStats",
    # Base classes
    "VisionProvider",
    # Manager
    "VisionManager",
    # Stub provider
    "DeepSeekStubProvider",
    "create_stub_provider",
    # DeepSeek provider
    "DeepSeekVisionProvider",
    "create_deepseek_provider",
    # OpenAI provider
    "OpenAIVisionProvider",
    "create_openai_provider",
    # Anthropic provider
    "AnthropicVisionProvider",
    "create_anthropic_provider",
    # Factory
    "create_vision_provider",
    "get_vision_provider",
    "get_available_providers",
    # Exceptions
    "VisionError",
    "VisionProviderError",
    "VisionInputError",
    # Resilience
    "ResilientVisionProvider",
    "RetryConfig",
    "CircuitBreakerConfig",
    "CircuitState",
    "CircuitOpenError",
    "ProviderMetrics",
    "create_resilient_provider",
]
