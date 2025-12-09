"""OpenAI Vision Provider.

Real implementation using OpenAI GPT-4o/GPT-4-Vision API for vision analysis.
Supports GPT-4o and GPT-4-turbo with vision capabilities.

API Documentation: https://platform.openai.com/docs/guides/vision
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
DEFAULT_SYSTEM_PROMPT = """You are an expert engineering drawing analyzer specializing in \
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


class OpenAIVisionProvider(VisionProvider):
    """
    OpenAI Vision provider using GPT-4o/GPT-4-Vision API.

    Supports multimodal analysis for engineering drawings.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        system_prompt: Optional[str] = None,
        timeout_seconds: float = 60.0,
        max_tokens: int = 2048,
        detail: str = "high",  # low, high, auto
    ):
        """
        Initialize OpenAI Vision provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: API base URL (can be overridden for Azure OpenAI)
            model: Model name (gpt-4o, gpt-4-turbo, gpt-4-vision-preview)
            system_prompt: Custom system prompt for analysis
            timeout_seconds: Request timeout
            max_tokens: Maximum response tokens
            detail: Image detail level (low=faster/cheaper, high=better quality)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise VisionProviderError(
                "openai", "API key required. Set OPENAI_API_KEY environment variable."
            )

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.detail = detail

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
        Analyze image using OpenAI Vision API.

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
                                    "url": f"data:{image_type};base64,{image_b64}",
                                    "detail": self.detail,
                                },
                            },
                            {
                                "type": "text",
                                "text": "Analyze this engineering drawing. Respond with JSON only.",
                            },
                        ],
                    },
                ],
                "max_tokens": self.max_tokens,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},  # Enforce JSON response
            }

            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )

            if response.status_code != 200:
                error_detail = response.text[:500]
                logger.error(f"OpenAI API error: {response.status_code} - {error_detail}")
                raise VisionProviderError(
                    "openai",
                    f"API request failed with status {response.status_code}: {error_detail}",
                )

            result = response.json()
            return self._parse_response(result)

        except httpx.TimeoutException:
            raise VisionProviderError("openai", f"Request timeout after {self.timeout_seconds}s")
        except httpx.RequestError as e:
            raise VisionProviderError("openai", f"Request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise VisionProviderError("openai", f"Invalid JSON response: {str(e)}")

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
        """Parse OpenAI API response into VisionDescription."""
        try:
            choices = result.get("choices", [])
            if not choices:
                raise VisionProviderError("openai", "No response choices returned")

            content = choices[0].get("message", {}).get("content", "")
            if not content:
                raise VisionProviderError("openai", "Empty response content")

            # Parse JSON response
            parsed = json.loads(content)

            return VisionDescription(
                summary=parsed.get("summary", "Analysis complete"),
                details=parsed.get("details", []),
                confidence=min(1.0, max(0.0, float(parsed.get("confidence", 0.9)))),
            )

        except json.JSONDecodeError:
            # Fallback for non-JSON response
            logger.warning("OpenAI returned non-JSON response, using fallback parsing")
            return VisionDescription(
                summary=content[:500] if content else "Analysis complete",
                details=[],
                confidence=0.7,
            )
        except Exception as e:
            logger.warning(f"Failed to parse OpenAI response: {e}")
            raise VisionProviderError("openai", f"Response parsing failed: {str(e)}")

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "openai"

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


def create_openai_provider(
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    detail: str = "high",
    **kwargs: Any,
) -> OpenAIVisionProvider:
    """
    Factory function to create OpenAI Vision provider.

    Args:
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        model: Model name (gpt-4o, gpt-4-turbo)
        detail: Image detail level (low, high, auto)
        **kwargs: Additional configuration options

    Returns:
        Configured OpenAIVisionProvider instance

    Example:
        >>> provider = create_openai_provider(model="gpt-4o", detail="high")
        >>> result = await provider.analyze_image(image_bytes)
    """
    return OpenAIVisionProvider(api_key=api_key, model=model, detail=detail, **kwargs)
