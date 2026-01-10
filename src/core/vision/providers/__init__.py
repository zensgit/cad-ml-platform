"""Vision providers package.

Provides multiple vision provider implementations:
- DeepSeekStubProvider: Stub for testing (no API key required)
- DeepSeekVisionProvider: Real DeepSeek API integration
- OpenAIVisionProvider: OpenAI GPT-4o/GPT-4-Vision integration
- AnthropicVisionProvider: Claude 3 family integration
- QwenVLProvider: Alibaba Qwen-VL (通义千问视觉) - Best for China users
- GLM4VProvider: Zhipu AI GLM-4V (智谱清言视觉) - China alternative
- DoubaoVisionProvider: ByteDance Doubao Vision (豆包视觉) - Cost-effective China option
"""

from .anthropic import AnthropicVisionProvider, create_anthropic_provider
from .deepseek import DeepSeekVisionProvider, create_deepseek_provider
from .deepseek_stub import DeepSeekStubProvider, create_stub_provider
from .doubao import DoubaoVisionProvider, create_doubao_provider
from .glm4v import GLM4VProvider, create_glm4v_provider
from .openai import OpenAIVisionProvider, create_openai_provider
from .qwen_vl import QwenVLProvider, create_qwen_vl_provider

__all__ = [
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
    # Qwen-VL provider (China)
    "QwenVLProvider",
    "create_qwen_vl_provider",
    # GLM-4V provider (China)
    "GLM4VProvider",
    "create_glm4v_provider",
    # Doubao Vision provider (China)
    "DoubaoVisionProvider",
    "create_doubao_provider",
]
