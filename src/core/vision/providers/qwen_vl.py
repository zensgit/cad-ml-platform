"""Qwen-VL (通义千问视觉) Provider.

Real implementation using Alibaba Cloud DashScope API for vision analysis.
Supports Qwen-VL-Max, Qwen-VL-Plus, and Qwen3-VL models.

Qwen-VL is the leading multimodal model in China with excellent OCR and
engineering drawing analysis capabilities.

API Documentation: https://help.aliyun.com/zh/model-studio/vision
OpenAI Compatible Mode: https://help.aliyun.com/zh/model-studio/qwen-vl-compatible-with-openai
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Dict, Optional

import httpx

from ..base import VisionDescription, VisionProvider, VisionProviderError

logger = logging.getLogger(__name__)

# Default system prompt optimized for engineering drawings (Chinese + English)
DEFAULT_SYSTEM_PROMPT = """你是一位专业的工程图纸分析专家，精通机械制造和CAD图纸。

请分析提供的图像并提取以下信息：
1. **概述**: 简洁描述图纸内容（零件类型、主要特征）
2. **详细信息**: 具体观察包括：
   - 尺寸信息（尺寸、公差）
   - 几何特征（孔、螺纹、倒角、圆角、槽）
   - 表面粗糙度要求（Ra、Rz值）
   - 材料规格（如可见）
   - 标题栏信息（图号、版本、日期）
   - GD&T（几何尺寸和公差）标注
   - 剖面图、详图标识

仅返回有效的JSON格式：
{
    "summary": "图纸的简要描述",
    "details": ["详细信息1", "详细信息2", ...],
    "confidence": 0.0-1.0
}

请准确且专业。如有不确定之处，请注明并降低置信度。"""

# English version for non-Chinese users
DEFAULT_SYSTEM_PROMPT_EN = """You are an expert engineering drawing analyzer specializing in \
mechanical and manufacturing drawings.

Analyze the provided image and extract:
1. **Summary**: A concise description of what the drawing shows (part type, main features)
2. **Details**: Specific observations including:
   - Dimensional information (sizes, tolerances)
   - Geometric features (holes, threads, chamfers, fillets)
   - Surface finish requirements (Ra values)
   - Material specifications if visible
   - Title block information (drawing number, revision, date)
   - GD&T (Geometric Dimensioning and Tolerancing) callouts
   - Section views, detail views identification

Respond ONLY with valid JSON:
{
    "summary": "Brief description of the drawing",
    "details": ["Detail 1", "Detail 2", ...],
    "confidence": 0.0-1.0
}

Be precise and technical. If something is unclear, note it with lower confidence."""


class QwenVLProvider(VisionProvider):
    """
    Qwen-VL (通义千问视觉) provider using Alibaba Cloud DashScope API.

    Supports OpenAI-compatible API mode for easy integration.
    Available models:
    - qwen-vl-max: Best quality, highest accuracy
    - qwen-vl-plus: Good balance of quality and speed
    - qwen3-vl-max: Latest generation, recommended
    - qwen-vl-ocr-latest: Optimized for OCR tasks
    """

    # DashScope API endpoints
    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DASHSCOPE_INTL_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "qwen-vl-max",
        system_prompt: Optional[str] = None,
        use_chinese_prompt: bool = True,
        timeout_seconds: float = 120.0,
        max_tokens: int = 2048,
        international: bool = False,
    ):
        """
        Initialize Qwen-VL Vision provider.

        Args:
            api_key: DashScope API key (defaults to DASHSCOPE_API_KEY env var)
            base_url: API base URL (defaults to DashScope endpoint)
            model: Model name (qwen-vl-max, qwen-vl-plus, qwen3-vl-max, qwen-vl-ocr-latest)
            system_prompt: Custom system prompt for analysis
            use_chinese_prompt: Whether to use Chinese system prompt (better for Chinese drawings)
            timeout_seconds: Request timeout (Qwen-VL may need longer for complex drawings)
            max_tokens: Maximum response tokens
            international: Use international endpoint (for users outside China)
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise VisionProviderError(
                "qwen_vl",
                "API key required. Set DASHSCOPE_API_KEY or QWEN_API_KEY environment variable.",
            )

        # Determine base URL
        if base_url:
            self.base_url = base_url.rstrip("/")
        elif international:
            self.base_url = self.DASHSCOPE_INTL_BASE_URL
        else:
            self.base_url = self.DASHSCOPE_BASE_URL

        self.model = model
        self.use_chinese_prompt = use_chinese_prompt

        if system_prompt:
            self.system_prompt = system_prompt
        elif use_chinese_prompt:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT
        else:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT_EN

        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens

        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True
    ) -> VisionDescription:
        """
        Analyze image using Qwen-VL API.

        Args:
            image_data: Raw image bytes
            include_description: Whether to generate detailed description

        Returns:
            VisionDescription with analysis results

        Raises:
            VisionProviderError: On API or processing errors
        """
        if not image_data:
            raise ValueError("image_data cannot be empty")

        if not include_description:
            return VisionDescription(
                summary="Image processed (OCR-only mode)",
                details=[],
                confidence=1.0,
            )

        try:
            # Encode image to base64
            image_b64 = base64.b64encode(image_data).decode("utf-8")

            # Detect image type
            image_type = self._detect_image_type(image_data)

            # Build request payload (OpenAI compatible format)
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image_type};base64,{image_b64}",
                                },
                            },
                            {
                                "type": "text",
                                "text": "请分析这张工程图纸，仅返回JSON格式。"
                                if self.use_chinese_prompt
                                else "Analyze this engineering drawing. Respond with JSON only.",
                            },
                        ],
                    },
                ],
                "max_tokens": self.max_tokens,
                "temperature": 0.1,
            }

            # Note: Qwen-VL doesn't always support response_format, so we rely on prompt engineering
            # Some models support it, add if available
            if self.model in ("qwen-vl-max", "qwen3-vl-max"):
                payload["response_format"] = {"type": "json_object"}

            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )

            if response.status_code != 200:
                error_detail = response.text[:500]
                logger.error(f"Qwen-VL API error: {response.status_code} - {error_detail}")
                raise VisionProviderError(
                    "qwen_vl",
                    f"API request failed with status {response.status_code}: {error_detail}",
                )

            result = response.json()
            return self._parse_response(result)

        except httpx.TimeoutException:
            raise VisionProviderError("qwen_vl", f"Request timeout after {self.timeout_seconds}s")
        except httpx.RequestError as e:
            raise VisionProviderError("qwen_vl", f"Request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise VisionProviderError("qwen_vl", f"Invalid JSON response: {str(e)}")

    def _detect_image_type(self, image_data: bytes) -> str:
        """Detect image MIME type from magic bytes."""
        if image_data[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        elif image_data[:2] == b"\xff\xd8":
            return "image/jpeg"
        elif image_data[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        elif image_data[:4] == b"RIFF" and image_data[8:12] == b"WEBP":
            return "image/webp"
        else:
            return "image/png"

    def _parse_response(self, result: Dict[str, Any]) -> VisionDescription:
        """Parse Qwen-VL API response into VisionDescription."""
        try:
            choices = result.get("choices", [])
            if not choices:
                raise VisionProviderError("qwen_vl", "No response choices returned")

            content = choices[0].get("message", {}).get("content", "")
            if not content:
                raise VisionProviderError("qwen_vl", "Empty response content")

            # Try to extract JSON from response
            # Sometimes the model wraps JSON in markdown code blocks
            json_content = self._extract_json(content)

            # Parse JSON response
            parsed = json.loads(json_content)

            return VisionDescription(
                summary=parsed.get("summary", parsed.get("概述", "Analysis complete")),
                details=parsed.get("details", parsed.get("详细信息", [])),
                confidence=min(
                    1.0, max(0.0, float(parsed.get("confidence", parsed.get("置信度", 0.9))))
                ),
            )

        except json.JSONDecodeError:
            # Fallback for non-JSON response
            logger.warning("Qwen-VL returned non-JSON response, using fallback parsing")
            return VisionDescription(
                summary=content[:500] if content else "Analysis complete",
                details=[],
                confidence=0.7,
            )
        except Exception as e:
            logger.warning(f"Failed to parse Qwen-VL response: {e}")
            raise VisionProviderError("qwen_vl", f"Response parsing failed: {str(e)}")

    def _extract_json(self, content: str) -> str:
        """Extract JSON from content that may include markdown code blocks."""
        content = content.strip()

        # Remove markdown code block wrapper if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        # Find JSON object boundaries
        start_idx = content.find("{")
        end_idx = content.rfind("}")

        if start_idx != -1 and end_idx != -1:
            return content[start_idx : end_idx + 1]

        return content

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "qwen_vl"

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


def create_qwen_vl_provider(
    api_key: Optional[str] = None,
    model: str = "qwen-vl-max",
    use_chinese_prompt: bool = True,
    international: bool = False,
    **kwargs: Any,
) -> QwenVLProvider:
    """
    Factory function to create Qwen-VL Vision provider.

    Args:
        api_key: DashScope API key (defaults to DASHSCOPE_API_KEY env var)
        model: Model name (qwen-vl-max, qwen-vl-plus, qwen3-vl-max)
        use_chinese_prompt: Use Chinese system prompt (recommended for Chinese drawings)
        international: Use international endpoint (for users outside China)
        **kwargs: Additional configuration options

    Returns:
        Configured QwenVLProvider instance

    Example:
        >>> provider = create_qwen_vl_provider(model="qwen-vl-max")
        >>> result = await provider.analyze_image(image_bytes)

    Available Models:
        - qwen-vl-max: Best quality, recommended for complex drawings
        - qwen-vl-plus: Faster, good for simple drawings
        - qwen3-vl-max: Latest generation
        - qwen-vl-ocr-latest: Optimized for OCR/text extraction

    Pricing (as of 2024):
        - qwen-vl-max: ¥0.02/千tokens
        - qwen-vl-plus: ¥0.008/千tokens
    """
    return QwenVLProvider(
        api_key=api_key,
        model=model,
        use_chinese_prompt=use_chinese_prompt,
        international=international,
        **kwargs,
    )
