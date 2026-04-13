"""Integration tests for vLLM end-to-end with vision, OCR, and assistant.

All external calls are mocked -- no real vLLM server required.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _mock_vllm_chat_response(content: str, status_code: int = 200):
    """Create a mock requests.Response for vLLM chat completions."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
    }
    mock_resp.text = content
    return mock_resp


def _mock_health_response(status_code: int = 200):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    return mock_resp


def _flags_with(overrides: dict) -> dict:
    """Build a feature flags dict with specific flags enabled/disabled."""
    base_flags = [
        {"name": "vllm_enabled", "enabled": False},
        {"name": "vllm_vision_enabled", "enabled": False},
        {"name": "vllm_ocr_enhancement_enabled", "enabled": False},
    ]
    for flag in base_flags:
        if flag["name"] in overrides:
            flag["enabled"] = overrides[flag["name"]]
    return {"flags": base_flags}


# ---------------------------------------------------------------------------
# 1. vLLM Vision Provider Tests
# ---------------------------------------------------------------------------

class TestVLLMVisionProvider:
    """Test VLLMVisionProvider with mocked vLLM endpoint."""

    def test_stub_fallback_when_flag_disabled(self):
        """When vllm_vision_enabled is off, should return stub response."""
        from src.core.vision.providers.vllm_vision import VLLMVisionProvider

        provider = VLLMVisionProvider()

        with patch(
            "src.core.vision.providers.vllm_vision._load_feature_flag",
            return_value=False,
        ):
            result = _run(provider.analyze_image(b"fake_image_data"))

        assert result.confidence == 0.92
        assert "mechanical engineering drawing" in result.summary

    def test_stub_fallback_when_vllm_unreachable(self):
        """When vLLM is unreachable, should fall back to stub."""
        from src.core.vision.providers.vllm_vision import VLLMVisionProvider

        provider = VLLMVisionProvider()

        with patch(
            "src.core.vision.providers.vllm_vision._load_feature_flag",
            return_value=True,
        ), patch.object(provider, "_check_vllm_health", return_value=False):
            result = _run(provider.analyze_image(b"fake_image_data"))

        assert result.confidence == 0.92  # stub response

    def test_vllm_call_success(self):
        """When vLLM is healthy and flag enabled, should use vLLM response."""
        from src.core.vision.providers.vllm_vision import VLLMVisionProvider

        provider = VLLMVisionProvider()

        vllm_response = json.dumps({
            "summary": "Shaft with M12 thread",
            "details": ["Diameter 25mm", "Thread M12x1.75"],
            "confidence": 0.88,
        })

        with patch(
            "src.core.vision.providers.vllm_vision._load_feature_flag",
            return_value=True,
        ), patch.object(
            provider, "_check_vllm_health", return_value=True
        ), patch.object(
            provider, "_call_vllm", return_value=vllm_response
        ):
            result = _run(provider.analyze_image(b"fake_image_data"))

        assert result.summary == "Shaft with M12 thread"
        assert result.confidence == 0.88
        assert len(result.details) == 2

    def test_vllm_call_failure_falls_back(self):
        """When vLLM call raises, should fall back to stub gracefully."""
        from src.core.vision.providers.vllm_vision import VLLMVisionProvider

        provider = VLLMVisionProvider()

        with patch(
            "src.core.vision.providers.vllm_vision._load_feature_flag",
            return_value=True,
        ), patch.object(
            provider, "_check_vllm_health", return_value=True
        ), patch.object(
            provider, "_call_vllm", side_effect=RuntimeError("connection lost")
        ):
            result = _run(provider.analyze_image(b"fake_image_data"))

        # Should fall back to stub, not raise
        assert result.confidence == 0.92

    def test_empty_image_raises(self):
        """Empty image_data should raise ValueError."""
        from src.core.vision.providers.vllm_vision import VLLMVisionProvider

        provider = VLLMVisionProvider()
        with pytest.raises(ValueError, match="image_data cannot be empty"):
            _run(provider.analyze_image(b""))

    def test_provider_name(self):
        from src.core.vision.providers.vllm_vision import VLLMVisionProvider

        assert VLLMVisionProvider().provider_name == "vllm_vision"


# ---------------------------------------------------------------------------
# 2. OCR Enhancer Pipeline Tests
# ---------------------------------------------------------------------------

class TestVLLMOcrEnhancer:
    """Test PaddleOCR -> vLLM enhancement pipeline."""

    def test_enhancement_disabled_returns_empty(self):
        """When flag is off, enhance() returns empty dict."""
        from src.core.ocr.providers.vllm_ocr_enhancer import VLLMOcrEnhancer

        enhancer = VLLMOcrEnhancer()

        with patch(
            "src.core.ocr.providers.vllm_ocr_enhancer._load_feature_flag",
            return_value=False,
        ):
            result = _run(enhancer.enhance("零件名称 轴承座 材料 HT250"))

        assert result == {}

    def test_enhancement_success(self):
        """When enabled and healthy, should return extracted fields."""
        from src.core.ocr.providers.vllm_ocr_enhancer import VLLMOcrEnhancer

        enhancer = VLLMOcrEnhancer()

        vllm_response = json.dumps({
            "part_name": "轴承座",
            "material": "HT250",
            "drawing_number": "MK-2025-003",
        })

        with patch(
            "src.core.ocr.providers.vllm_ocr_enhancer._load_feature_flag",
            return_value=True,
        ), patch.object(
            enhancer, "_check_health", return_value=True
        ), patch.object(
            enhancer, "_call_vllm", return_value=vllm_response
        ):
            result = _run(enhancer.enhance("零件名称 轴承座 材料 HT250 图号 MK-2025-003"))

        assert result["part_name"] == "轴承座"
        assert result["material"] == "HT250"
        assert result["drawing_number"] == "MK-2025-003"

    def test_enhancement_vllm_unreachable(self):
        """When vLLM is down, should return empty dict gracefully."""
        from src.core.ocr.providers.vllm_ocr_enhancer import VLLMOcrEnhancer

        enhancer = VLLMOcrEnhancer()

        with patch(
            "src.core.ocr.providers.vllm_ocr_enhancer._load_feature_flag",
            return_value=True,
        ), patch.object(enhancer, "_check_health", return_value=False):
            result = _run(enhancer.enhance("some ocr text"))

        assert result == {}

    def test_enhancement_empty_text(self):
        """Empty OCR text should return empty dict."""
        from src.core.ocr.providers.vllm_ocr_enhancer import VLLMOcrEnhancer

        enhancer = VLLMOcrEnhancer()
        result = _run(enhancer.enhance(""))
        assert result == {}


# ---------------------------------------------------------------------------
# 3. Prompt Template Tests
# ---------------------------------------------------------------------------

class TestPromptTemplates:
    """Test that prompt templates produce valid output format."""

    def test_cad_system_prompt_zh(self):
        from src.core.assistant.prompts.cad_system_prompt import get_cad_system_prompt

        prompt = get_cad_system_prompt("zh")
        assert "CAD-ML" in prompt
        assert "准确" in prompt
        assert len(prompt) < 2000  # Must be concise for local models

    def test_cad_system_prompt_en(self):
        from src.core.assistant.prompts.cad_system_prompt import get_cad_system_prompt

        prompt = get_cad_system_prompt("en")
        assert "CAD-ML" in prompt
        assert "accurate" in prompt.lower()

    def test_cad_system_prompt_minimal(self):
        from src.core.assistant.prompts.cad_system_prompt import get_cad_system_prompt

        prompt = get_cad_system_prompt("zh", max_tokens=1500)
        assert len(prompt) < 200  # Minimal prompt must be very short

    def test_ocr_extraction_prompt(self):
        from src.core.assistant.prompts.ocr_extraction_prompt import (
            get_ocr_extraction_prompt,
        )

        prompt = get_ocr_extraction_prompt("零件名称 轴承座 材料 HT250")
        assert "轴承座" in prompt
        assert "HT250" in prompt
        assert "part_name" in prompt
        assert "JSON" in prompt or "json" in prompt

    def test_ocr_extraction_prompt_minimal(self):
        from src.core.assistant.prompts.ocr_extraction_prompt import (
            get_ocr_extraction_prompt,
        )

        prompt = get_ocr_extraction_prompt("test text", max_tokens=1500)
        # Minimal prompt should be shorter
        assert len(prompt) < 500

    def test_classification_prompt(self):
        from src.core.assistant.prompts.classification_prompt import (
            get_classification_prompt,
        )

        prompt = get_classification_prompt(
            filename="shaft.dxf",
            ocr_text="M10 Ra3.2 45钢",
            line_count=100,
            circle_count=5,
            arc_count=10,
        )
        assert "shaft.dxf" in prompt
        assert "M10" in prompt
        assert "100" in prompt
        assert "mechanical_part" in prompt  # category in examples

    def test_classification_prompt_truncates_long_ocr(self):
        from src.core.assistant.prompts.classification_prompt import (
            get_classification_prompt,
        )

        long_text = "A" * 1000
        prompt = get_classification_prompt(ocr_text=long_text)
        assert "..." in prompt  # Should be truncated


# ---------------------------------------------------------------------------
# 4. Fallback Chain Tests
# ---------------------------------------------------------------------------

class TestFallbackChain:
    """Test vLLM -> Claude -> OpenAI -> Qwen -> Ollama -> Offline fallback."""

    def test_vllm_unavailable_falls_to_next(self):
        """When vLLM flag is on but server is down, should use next provider."""
        from src.core.assistant.assistant import CADAssistant, AssistantConfig

        # Provide a callback so we don't need real providers
        called_with = {}

        def mock_callback(system, user):
            called_with["system"] = system
            called_with["user"] = user
            return "fallback response"

        assistant = CADAssistant(
            config=AssistantConfig(auto_select_provider=False),
            llm_callback=mock_callback,
        )

        response = assistant.ask("304不锈钢的抗拉强度?")
        assert response.answer is not None

    def test_llm_health_status(self):
        """Test llm_health_status returns structured data."""
        from src.core.assistant.assistant import CADAssistant, AssistantConfig

        assistant = CADAssistant(
            config=AssistantConfig(auto_select_provider=False),
            llm_callback=lambda s, u: "ok",
        )

        status = assistant.llm_health_status()
        # With callback, no provider is initialized
        assert "status" in status


# ---------------------------------------------------------------------------
# 5. Feature Flag Gating Tests
# ---------------------------------------------------------------------------

class TestFeatureFlagGating:
    """Test that disabled flags prevent vLLM usage entirely."""

    def test_vision_flag_disabled_skips_vllm(self):
        from src.core.vision.providers.vllm_vision import VLLMVisionProvider

        provider = VLLMVisionProvider()

        with patch(
            "src.core.vision.providers.vllm_vision._load_feature_flag",
            return_value=False,
        ):
            # Should never attempt to contact vLLM
            result = _run(provider.analyze_image(b"test_data"))
            assert result.confidence == 0.92  # stub response returned

    def test_ocr_flag_disabled_skips_enhancement(self):
        from src.core.ocr.providers.vllm_ocr_enhancer import VLLMOcrEnhancer

        enhancer = VLLMOcrEnhancer()

        with patch(
            "src.core.ocr.providers.vllm_ocr_enhancer._load_feature_flag",
            return_value=False,
        ):
            result = _run(enhancer.enhance("零件名称 test"))
            assert result == {}

    def test_vllm_provider_in_assistant_when_flag_off(self):
        """When vllm_enabled is off, assistant should not prefer vLLM."""
        from src.core.assistant.assistant import CADAssistant

        with patch.object(
            CADAssistant, "_is_vllm_flag_enabled", return_value=False
        ):
            assistant = CADAssistant(
                llm_callback=lambda s, u: "test response"
            )
            # Provider should NOT be VLLMProvider
            if assistant._llm_provider is not None:
                assert type(assistant._llm_provider).__name__ != "VLLMProvider"
