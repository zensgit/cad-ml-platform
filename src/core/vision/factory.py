"""Vision Provider Factory.

Centralized factory for creating and managing vision providers.
Supports automatic provider selection based on configuration.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from .base import VisionProvider, VisionProviderError
from .providers import (
    AnthropicVisionProvider,
    DeepSeekStubProvider,
    DeepSeekVisionProvider,
    DoubaoVisionProvider,
    GLM4VProvider,
    OpenAIVisionProvider,
    QwenVLProvider,
    create_anthropic_provider,
    create_deepseek_provider,
    create_doubao_provider,
    create_glm4v_provider,
    create_openai_provider,
    create_qwen_vl_provider,
    create_stub_provider,
)

logger = logging.getLogger(__name__)

# Provider registry
PROVIDER_REGISTRY: Dict[str, type] = {
    "stub": DeepSeekStubProvider,
    "deepseek_stub": DeepSeekStubProvider,
    "deepseek": DeepSeekVisionProvider,
    "openai": OpenAIVisionProvider,
    "gpt4o": OpenAIVisionProvider,
    "gpt-4o": OpenAIVisionProvider,
    "anthropic": AnthropicVisionProvider,
    "claude": AnthropicVisionProvider,
    # Qwen-VL (通义千问) - Best for China users
    "qwen": QwenVLProvider,
    "qwen_vl": QwenVLProvider,
    "qwen-vl": QwenVLProvider,
    "tongyi": QwenVLProvider,
    "dashscope": QwenVLProvider,
    # GLM-4V (智谱清言) - China alternative
    "glm": GLM4VProvider,
    "glm4v": GLM4VProvider,
    "glm-4v": GLM4VProvider,
    "zhipu": GLM4VProvider,
    "chatglm": GLM4VProvider,
    # Doubao Vision (豆包视觉) - Cost-effective China option
    "doubao": DoubaoVisionProvider,
    "doubao_vision": DoubaoVisionProvider,
    "doubao-vision": DoubaoVisionProvider,
    "bytedance": DoubaoVisionProvider,
    "volcengine": DoubaoVisionProvider,
    "ark": DoubaoVisionProvider,
}

# Factory function registry
FACTORY_REGISTRY: Dict[str, callable] = {
    "stub": create_stub_provider,
    "deepseek_stub": create_stub_provider,
    "deepseek": create_deepseek_provider,
    "openai": create_openai_provider,
    "gpt4o": create_openai_provider,
    "gpt-4o": create_openai_provider,
    "anthropic": create_anthropic_provider,
    "claude": create_anthropic_provider,
    # Qwen-VL (通义千问) - Best for China users
    "qwen": create_qwen_vl_provider,
    "qwen_vl": create_qwen_vl_provider,
    "qwen-vl": create_qwen_vl_provider,
    "tongyi": create_qwen_vl_provider,
    "dashscope": create_qwen_vl_provider,
    # GLM-4V (智谱清言) - China alternative
    "glm": create_glm4v_provider,
    "glm4v": create_glm4v_provider,
    "glm-4v": create_glm4v_provider,
    "zhipu": create_glm4v_provider,
    "chatglm": create_glm4v_provider,
    # Doubao Vision (豆包视觉) - Cost-effective China option
    "doubao": create_doubao_provider,
    "doubao_vision": create_doubao_provider,
    "doubao-vision": create_doubao_provider,
    "bytedance": create_doubao_provider,
    "volcengine": create_doubao_provider,
    "ark": create_doubao_provider,
}

# Default model configurations per provider
DEFAULT_MODELS: Dict[str, str] = {
    "deepseek": "deepseek-chat",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "qwen": "qwen-vl-max",
    "qwen_vl": "qwen-vl-max",
    "qwen-vl": "qwen-vl-max",
    "tongyi": "qwen-vl-max",
    "dashscope": "qwen-vl-max",
    "glm": "glm-4v-plus",
    "glm4v": "glm-4v-plus",
    "glm-4v": "glm-4v-plus",
    "zhipu": "glm-4v-plus",
    "chatglm": "glm-4v-plus",
    "doubao": "doubao-1-5-vision-pro-32k-250115",
    "doubao_vision": "doubao-1-5-vision-pro-32k-250115",
    "doubao-vision": "doubao-1-5-vision-pro-32k-250115",
    "bytedance": "doubao-1-5-vision-pro-32k-250115",
    "volcengine": "doubao-1-5-vision-pro-32k-250115",
    "ark": "doubao-1-5-vision-pro-32k-250115",
}


def create_vision_provider(
    provider_type: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    fallback_to_stub: bool = True,
    **kwargs: Any,
) -> VisionProvider:
    """
    Create a vision provider based on configuration.

    Provider selection priority:
    1. Explicit provider_type parameter
    2. VISION_PROVIDER environment variable
    3. Auto-detect based on available API keys
    4. Fallback to stub (if fallback_to_stub=True)

    Args:
        provider_type: Provider type (deepseek, openai, anthropic, stub)
        api_key: API key (overrides environment variable)
        model: Model name (overrides default)
        fallback_to_stub: Whether to fallback to stub if real provider fails
        **kwargs: Additional provider-specific options

    Returns:
        Configured VisionProvider instance

    Raises:
        VisionProviderError: If no provider can be created

    Examples:
        >>> # Auto-detect provider
        >>> provider = create_vision_provider()

        >>> # Explicit provider selection
        >>> provider = create_vision_provider("openai", model="gpt-4o")

        >>> # With custom API key
        >>> provider = create_vision_provider("deepseek", api_key="sk-...")
    """
    # Determine provider type
    provider = provider_type or os.getenv("VISION_PROVIDER", "").lower()

    # Auto-detect if not specified
    if not provider or provider == "auto":
        provider = _auto_detect_provider()

    provider = provider.lower()

    # Check if provider is in registry
    if provider not in FACTORY_REGISTRY:
        available = ", ".join(sorted(set(FACTORY_REGISTRY.keys())))
        raise VisionProviderError(
            "factory",
            f"Unknown provider '{provider}'. Available: {available}",
        )

    # Get factory function
    factory_fn = FACTORY_REGISTRY[provider]

    # Build kwargs
    provider_kwargs = dict(kwargs)
    if api_key:
        provider_kwargs["api_key"] = api_key
    if model:
        provider_kwargs["model"] = model
    elif provider in DEFAULT_MODELS:
        provider_kwargs.setdefault("model", DEFAULT_MODELS[provider])

    # Try to create provider
    try:
        instance = factory_fn(**provider_kwargs)
        logger.info(f"Created vision provider: {instance.provider_name}")
        return instance

    except VisionProviderError as e:
        if fallback_to_stub and provider != "stub":
            logger.warning(f"Failed to create {provider} provider: {e}. Falling back to stub.")
            return create_stub_provider()
        raise

    except Exception as e:
        if fallback_to_stub and provider != "stub":
            logger.warning(
                f"Unexpected error creating {provider} provider: {e}. Falling back to stub."
            )
            return create_stub_provider()
        raise VisionProviderError("factory", f"Failed to create provider: {str(e)}")


def _auto_detect_provider() -> str:
    """
    Auto-detect which provider to use based on available API keys.

    Detection priority:
    1. DASHSCOPE_API_KEY or QWEN_API_KEY -> qwen_vl (recommended for China)
    2. ZHIPUAI_API_KEY or GLM_API_KEY -> glm4v (China alternative)
    3. ARK_API_KEY or VOLCENGINE_API_KEY -> doubao (cost-effective China option)
    4. DEEPSEEK_API_KEY -> deepseek
    5. OPENAI_API_KEY -> openai
    6. ANTHROPIC_API_KEY -> anthropic
    7. Default -> stub
    """
    # Qwen-VL first (best for China users)
    if os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY"):
        logger.info("Auto-detected Qwen-VL provider (DASHSCOPE_API_KEY/QWEN_API_KEY present)")
        return "qwen_vl"

    # GLM-4V second (China alternative)
    if os.getenv("ZHIPUAI_API_KEY") or os.getenv("GLM_API_KEY") or os.getenv("ZHIPU_API_KEY"):
        logger.info("Auto-detected GLM-4V provider (ZHIPUAI_API_KEY/GLM_API_KEY present)")
        return "glm4v"

    # Doubao Vision third (cost-effective China option)
    if os.getenv("ARK_API_KEY") or os.getenv("VOLCENGINE_API_KEY") or os.getenv("DOUBAO_API_KEY"):
        logger.info("Auto-detected Doubao Vision provider (ARK_API_KEY/VOLCENGINE_API_KEY present)")
        return "doubao"

    if os.getenv("DEEPSEEK_API_KEY"):
        logger.info("Auto-detected DeepSeek provider (DEEPSEEK_API_KEY present)")
        return "deepseek"

    if os.getenv("OPENAI_API_KEY"):
        logger.info("Auto-detected OpenAI provider (OPENAI_API_KEY present)")
        return "openai"

    if os.getenv("ANTHROPIC_API_KEY"):
        logger.info("Auto-detected Anthropic provider (ANTHROPIC_API_KEY present)")
        return "anthropic"

    logger.info("No API keys found, defaulting to stub provider")
    return "stub"


def get_available_providers() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available providers and their status.

    Returns:
        Dict mapping provider names to their availability status

    Example:
        >>> providers = get_available_providers()
        >>> print(providers)
        {
            "stub": {"available": True, "requires_key": False},
            "deepseek": {"available": True, "requires_key": True, "key_set": True},
            "openai": {"available": True, "requires_key": True, "key_set": False},
            ...
        }
    """
    return {
        "stub": {
            "available": True,
            "requires_key": False,
            "description": "Stub provider for testing (returns fixed responses)",
        },
        "qwen_vl": {
            "available": True,
            "requires_key": True,
            "key_set": bool(os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")),
            "env_var": "DASHSCOPE_API_KEY or QWEN_API_KEY",
            "default_model": DEFAULT_MODELS.get("qwen_vl"),
            "description": "Alibaba Qwen-VL (通义千问视觉) - Best for China users, excellent OCR",
            "aliases": ["qwen", "qwen-vl", "tongyi", "dashscope"],
        },
        "glm4v": {
            "available": True,
            "requires_key": True,
            "key_set": bool(
                os.getenv("ZHIPUAI_API_KEY")
                or os.getenv("GLM_API_KEY")
                or os.getenv("ZHIPU_API_KEY")
            ),
            "env_var": "ZHIPUAI_API_KEY, GLM_API_KEY, or ZHIPU_API_KEY",
            "default_model": DEFAULT_MODELS.get("glm4v"),
            "description": "Zhipu AI GLM-4V (智谱清言视觉) - China alternative, free tier available",
            "aliases": ["glm", "glm-4v", "zhipu", "chatglm"],
        },
        "doubao": {
            "available": True,
            "requires_key": True,
            "key_set": bool(
                os.getenv("ARK_API_KEY")
                or os.getenv("VOLCENGINE_API_KEY")
                or os.getenv("DOUBAO_API_KEY")
            ),
            "env_var": "ARK_API_KEY, VOLCENGINE_API_KEY, or DOUBAO_API_KEY",
            "default_model": DEFAULT_MODELS.get("doubao"),
            "description": "ByteDance Doubao Vision (豆包视觉) - Cost-effective, ¥0.003/千tokens",
            "aliases": ["doubao_vision", "doubao-vision", "bytedance", "volcengine", "ark"],
        },
        "deepseek": {
            "available": True,
            "requires_key": True,
            "key_set": bool(os.getenv("DEEPSEEK_API_KEY")),
            "env_var": "DEEPSEEK_API_KEY",
            "default_model": DEFAULT_MODELS.get("deepseek"),
            "description": "DeepSeek VL API for vision analysis",
        },
        "openai": {
            "available": True,
            "requires_key": True,
            "key_set": bool(os.getenv("OPENAI_API_KEY")),
            "env_var": "OPENAI_API_KEY",
            "default_model": DEFAULT_MODELS.get("openai"),
            "description": "OpenAI GPT-4o/GPT-4-Vision for vision analysis",
        },
        "anthropic": {
            "available": True,
            "requires_key": True,
            "key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
            "env_var": "ANTHROPIC_API_KEY",
            "default_model": DEFAULT_MODELS.get("anthropic"),
            "description": "Anthropic Claude 3 family for vision analysis",
        },
    }


# Convenience alias
get_vision_provider = create_vision_provider
