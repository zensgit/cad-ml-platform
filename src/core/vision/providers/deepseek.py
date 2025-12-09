"""DeepSeek Vision Provider.

Real implementation using DeepSeek VL API for vision analysis.
Supports DeepSeek-VL2 multimodal model for engineering drawing understanding.

API Documentation: https://platform.deepseek.com/api-docs
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

# Default system prompt optimized for engineering drawings
DEFAULT_SYSTEM_PROMPT = """You are an expert engineering drawing analyzer. \
Analyze the provided image and extract:

1. **Summary**: A concise description of what the drawing shows (part type, main features)
2. **Details**: Specific observations including:
   - Dimensional information (sizes, tolerances)
   - Geometric features (holes, threads, chamfers, fillets)
   - Surface finish requirements
   - Material specifications if visible
   - Title block information
   - Any GD&T (Geometric Dimensioning and Tolerancing) callouts

Respond in JSON format:
{
    "summary": "Brief description of the drawing",
    "details": ["Detail 1", "Detail 2", ...],
    "confidence": 0.0-1.0 (your confidence in the analysis)
}

Focus on accuracy. If something is unclear, note it in the details."""


class DeepSeekVisionProvider(VisionProvider):
    """
    DeepSeek Vision provider using the DeepSeek API.

    Supports DeepSeek-VL2 multimodal model for analyzing engineering drawings.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        system_prompt: Optional[str] = None,
        timeout_seconds: float = 60.0,
        max_tokens: int = 2048,
    ):
        """
        Initialize DeepSeek Vision provider.

        Args:
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
            base_url: API base URL
            model: Model name (deepseek-chat supports vision)
            system_prompt: Custom system prompt for analysis
            timeout_seconds: Request timeout
            max_tokens: Maximum response tokens
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise VisionProviderError(
                "deepseek", "API key required. Set DEEPSEEK_API_KEY environment variable."
            )

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
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
        Analyze image using DeepSeek Vision API.

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

            # Build request payload
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
                                    "url": f"data:{image_type};base64,{image_b64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": "Analyze this engineering drawing. Respond JSON only.",
                            },
                        ],
                    },
                ],
                "max_tokens": self.max_tokens,
                "temperature": 0.1,  # Low temperature for consistent analysis
            }

            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )

            if response.status_code != 200:
                error_detail = response.text[:500]
                logger.error(f"DeepSeek API error: {response.status_code} - {error_detail}")
                raise VisionProviderError(
                    "deepseek",
                    f"API request failed with status {response.status_code}: {error_detail}",
                )

            result = response.json()
            return self._parse_response(result)

        except httpx.TimeoutException:
            raise VisionProviderError("deepseek", f"Request timeout after {self.timeout_seconds}s")
        except httpx.RequestError as e:
            raise VisionProviderError("deepseek", f"Request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise VisionProviderError("deepseek", f"Invalid JSON response: {str(e)}")

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
            return "image/png"  # Default fallback

    def _parse_response(self, result: Dict[str, Any]) -> VisionDescription:
        """Parse DeepSeek API response into VisionDescription."""
        try:
            choices = result.get("choices", [])
            if not choices:
                raise VisionProviderError("deepseek", "No response choices returned")

            content = choices[0].get("message", {}).get("content", "")
            if not content:
                raise VisionProviderError("deepseek", "Empty response content")

            # Try to parse as JSON
            parsed = self._extract_json(content)

            if parsed:
                return VisionDescription(
                    summary=parsed.get("summary", "Analysis complete"),
                    details=parsed.get("details", []),
                    confidence=min(1.0, max(0.0, float(parsed.get("confidence", 0.85)))),
                )
            else:
                # Fallback: treat entire response as summary
                return VisionDescription(
                    summary=content[:500],
                    details=[content] if len(content) > 500 else [],
                    confidence=0.7,  # Lower confidence for non-JSON response
                )

        except Exception as e:
            logger.warning(f"Failed to parse DeepSeek response: {e}")
            raise VisionProviderError("deepseek", f"Response parsing failed: {str(e)}")

    def _extract_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from response content."""
        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code block
        import re

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        json_match = re.search(r"\{[^{}]*\"summary\"[^{}]*\}", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "deepseek"

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


def create_deepseek_provider(
    api_key: Optional[str] = None,
    model: str = "deepseek-chat",
    **kwargs: Any,
) -> DeepSeekVisionProvider:
    """
    Factory function to create DeepSeek Vision provider.

    Args:
        api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
        model: Model name
        **kwargs: Additional configuration options

    Returns:
        Configured DeepSeekVisionProvider instance

    Example:
        >>> provider = create_deepseek_provider()
        >>> result = await provider.analyze_image(image_bytes)
    """
    return DeepSeekVisionProvider(api_key=api_key, model=model, **kwargs)
