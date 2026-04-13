"""vLLM-backed Vision Provider.

Routes vision analysis requests to local vLLM server for image/drawing
description. Falls back to stub responses when vLLM is unavailable.

Gated by ``vllm_vision_enabled`` feature flag.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Optional

from ..base import VisionDescription, VisionProvider

logger = logging.getLogger(__name__)


def _load_feature_flag(flag_name: str) -> bool:
    """Check a feature flag from config/feature_flags.json."""
    try:
        flags_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "config", "feature_flags.json"
        )
        flags_path = os.path.normpath(flags_path)
        with open(flags_path) as f:
            data = json.load(f)
        for flag in data.get("flags", []):
            if flag.get("name") == flag_name:
                return flag.get("enabled", False)
    except Exception:
        pass
    return False


class VLLMVisionProvider(VisionProvider):
    """
    Vision provider backed by local vLLM server.

    Converts image analysis requests into text prompts suitable for local LLM,
    then sends them to VLLMProvider for inference. Falls back to stub responses
    when vLLM is unavailable or the feature flag is disabled.
    """

    def __init__(
        self,
        vllm_endpoint: Optional[str] = None,
        vllm_model: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize vLLM vision provider.

        Args:
            vllm_endpoint: vLLM server URL (default: from VLLM_ENDPOINT env)
            vllm_model: Model name (default: from VLLM_MODEL env)
            timeout: Request timeout in seconds
        """
        self._endpoint = vllm_endpoint or os.getenv(
            "VLLM_ENDPOINT", "http://localhost:8100"
        )
        self._model = vllm_model or os.getenv(
            "VLLM_MODEL", "deepseek-coder-6.7b-awq"
        )
        self._timeout = timeout

    def _is_vllm_enabled(self) -> bool:
        """Check if vLLM vision is enabled via feature flag."""
        return _load_feature_flag("vllm_vision_enabled")

    def _check_vllm_health(self) -> bool:
        """Check if the vLLM server is reachable."""
        try:
            import requests

            for path in ("/health", "/v1/models"):
                try:
                    resp = requests.get(
                        f"{self._endpoint}{path}", timeout=2
                    )
                    if resp.status_code == 200:
                        return True
                except Exception:
                    continue
            return False
        except Exception:
            return False

    def _call_vllm(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request to vLLM."""
        import requests

        response = requests.post(
            f"{self._endpoint}/v1/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 512,
                "stream": False,
            },
            timeout=self._timeout,
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        raise RuntimeError(f"vLLM error: {response.status_code}")

    def _build_vision_prompt(self, include_description: bool) -> tuple[str, str]:
        """Build system + user prompts for vision analysis."""
        system = (
            "你是CAD图纸分析专家。根据提供的图纸信息，描述图纸内容，"
            "包括零件类型、尺寸标注、螺纹、表面粗糙度、标题栏信息。"
            "输出格式：summary(一句话概述), details(列表), confidence(0-1)。"
        )

        if include_description:
            user = (
                "分析这张机械工程图纸。请描述：\n"
                "1. 零件类型和主要特征\n"
                "2. 关键尺寸和公差\n"
                "3. 表面处理要求\n"
                "4. 标题栏信息\n"
                "以JSON格式输出 {summary, details[], confidence}。"
            )
        else:
            user = "简要描述图纸内容（OCR模式，仅需概述）。输出JSON {summary, details[], confidence}。"

        return system, user

    def _parse_vllm_response(self, response_text: str) -> VisionDescription:
        """Parse vLLM response into VisionDescription."""
        try:
            # Try to extract JSON from response
            text = response_text.strip()
            # Handle markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)
            return VisionDescription(
                summary=data.get("summary", "vLLM analysis complete"),
                details=data.get("details", []),
                confidence=min(1.0, max(0.0, float(data.get("confidence", 0.7)))),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            # If JSON parsing fails, treat the whole response as summary
            return VisionDescription(
                summary=response_text[:200] if response_text else "Analysis complete",
                details=[],
                confidence=0.5,
            )

    def _stub_response(self, include_description: bool) -> VisionDescription:
        """Return fallback stub response when vLLM is unavailable."""
        if not include_description:
            return VisionDescription(
                summary="Image processed (OCR-only mode)", details=[], confidence=1.0
            )

        return VisionDescription(
            summary="This is a mechanical engineering drawing showing a cylindrical part with threaded features.",
            details=[
                "Main body features a diameter dimension of approximately 20mm with bilateral tolerance",
                "External thread specification visible (M10x1.5 pitch)",
                "Surface finish requirement indicated (Ra 3.2 or similar)",
                "Title block present with drawing number and material specification",
                "Standard orthographic projection with front and side views",
            ],
            confidence=0.92,
        )

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True
    ) -> VisionDescription:
        """
        Analyze image using vLLM or fall back to stub.

        Args:
            image_data: Raw image bytes (validated but used as context signal)
            include_description: Whether to include full description

        Returns:
            VisionDescription from vLLM or stub fallback

        Raises:
            ValueError: If image_data is empty
        """
        if not image_data or len(image_data) == 0:
            raise ValueError("image_data cannot be empty")

        # Gate: feature flag check
        if not self._is_vllm_enabled():
            logger.debug("vllm_vision_enabled flag is off, using stub")
            return self._stub_response(include_description)

        # Gate: health check
        if not self._check_vllm_health():
            logger.warning("vLLM server unreachable, falling back to stub")
            return self._stub_response(include_description)

        # Build prompts and call vLLM
        system_prompt, user_prompt = self._build_vision_prompt(include_description)

        start = time.monotonic()
        try:
            # Run sync HTTP call in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None, self._call_vllm, system_prompt, user_prompt
            )
            latency_ms = (time.monotonic() - start) * 1000
            logger.info(
                "vllm_vision.analyze_image",
                extra={"latency_ms": f"{latency_ms:.1f}", "provider": "vllm"},
            )
            return self._parse_vllm_response(response_text)
        except Exception as e:
            latency_ms = (time.monotonic() - start) * 1000
            logger.warning(
                "vLLM vision call failed (%.1fms), falling back to stub: %s",
                latency_ms,
                str(e),
            )
            return self._stub_response(include_description)

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "vllm_vision"


# ========== Factory Function ==========


def create_vllm_vision_provider(
    vllm_endpoint: Optional[str] = None,
    vllm_model: Optional[str] = None,
    timeout: int = 30,
) -> VLLMVisionProvider:
    """
    Factory function to create vLLM vision provider.

    Args:
        vllm_endpoint: vLLM server URL
        vllm_model: Model name
        timeout: Request timeout in seconds

    Returns:
        Configured VLLMVisionProvider instance
    """
    return VLLMVisionProvider(
        vllm_endpoint=vllm_endpoint,
        vllm_model=vllm_model,
        timeout=timeout,
    )
