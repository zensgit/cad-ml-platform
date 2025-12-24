"""Anthropic Vision Provider.

Real implementation using Claude API for vision analysis.
Supports Claude 3 Opus, Claude 3.5 Sonnet, Claude 3 Haiku with vision capabilities.

API Documentation: https://docs.anthropic.com/en/docs/vision
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
DEFAULT_SYSTEM_PROMPT = """You are an expert engineering drawing analyzer with deep knowledge of \
mechanical design, manufacturing processes, and technical documentation standards.

When analyzing engineering drawings, focus on:
1. **Summary**: Concise description of the part/assembly type and main features
2. **Details**: Technical observations including:
   - Dimensional information with tolerances
   - Geometric features (holes, threads, chamfers, fillets, slots)
   - Surface finish requirements (Ra, Rz values)
   - Material specifications
   - Title block data (drawing number, revision, scale, date)
   - GD&T callouts (datums, position, flatness, perpendicularity, etc.)
   - Section views and detail view identification
   - Bill of Materials if present

CRITICAL: Respond ONLY with valid JSON in this exact format:
{
    "summary": "Brief technical description",
    "details": ["Observation 1", "Observation 2", ...],
    "confidence": 0.85
}

Be precise and use proper engineering terminology. \
Estimate your confidence based on image clarity and completeness."""


class AnthropicVisionProvider(VisionProvider):
    """
    Anthropic Vision provider using Claude API.

    Supports Claude 3 family models (Opus, Sonnet, Haiku) for multimodal analysis.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com",
        model: str = "claude-sonnet-4-20250514",
        system_prompt: Optional[str] = None,
        timeout_seconds: float = 90.0,
        max_tokens: int = 2048,
    ):
        """
        Initialize Anthropic Vision provider.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            base_url: API base URL
            model: Model name (claude-sonnet-4-20250514, claude-3-5-sonnet-20241022, etc.)
            system_prompt: Custom system prompt for analysis
            timeout_seconds: Request timeout (Claude can be slower)
            max_tokens: Maximum response tokens
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise VisionProviderError(
                "anthropic", "API key required. Set ANTHROPIC_API_KEY environment variable."
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
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
            )
        return self._client

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True
    ) -> VisionDescription:
        """
        Analyze image using Anthropic Claude API.

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

            # Detect image type (Claude requires specific media types)
            media_type = self._detect_media_type(image_data)

            # Build request payload (Claude's message format)
            payload = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": self.system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": "Analyze this engineering drawing. Respond with JSON only.",
                            },
                        ],
                    }
                ],
            }

            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/v1/messages",
                json=payload,
            )

            if response.status_code != 200:
                error_detail = response.text[:500]
                logger.error(f"Anthropic API error: {response.status_code} - {error_detail}")
                raise VisionProviderError(
                    "anthropic",
                    f"API request failed with status {response.status_code}: {error_detail}",
                )

            result = response.json()
            return self._parse_response(result)

        except httpx.TimeoutException:
            raise VisionProviderError("anthropic", f"Request timeout after {self.timeout_seconds}s")
        except httpx.RequestError as e:
            raise VisionProviderError("anthropic", f"Request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise VisionProviderError("anthropic", f"Invalid JSON response: {str(e)}")

    def _detect_media_type(self, image_data: bytes) -> str:
        """Detect image media type from magic bytes (Claude-compatible types)."""
        if image_data[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        elif image_data[:2] == b"\xff\xd8":
            return "image/jpeg"
        elif image_data[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        elif image_data[:4] == b"RIFF" and image_data[8:12] == b"WEBP":
            return "image/webp"
        else:
            # Claude supports png, jpeg, gif, webp
            return "image/png"

    def _parse_response(self, result: Dict[str, Any]) -> VisionDescription:
        """Parse Anthropic API response into VisionDescription."""
        try:
            # Claude returns content as array of blocks
            content_blocks = result.get("content", [])
            if not content_blocks:
                raise VisionProviderError("anthropic", "No content in response")

            # Extract text from content blocks
            text_content = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    text_content += block.get("text", "")

            if not text_content:
                raise VisionProviderError("anthropic", "No text content in response")

            # Try to parse as JSON
            parsed = self._extract_json(text_content)

            if parsed:
                return VisionDescription(
                    summary=parsed.get("summary", "Analysis complete"),
                    details=parsed.get("details", []),
                    confidence=min(1.0, max(0.0, float(parsed.get("confidence", 0.9)))),
                )
            else:
                # Fallback: use raw text
                logger.warning("Anthropic returned non-JSON response, using fallback")
                return VisionDescription(
                    summary=text_content[:500],
                    details=[text_content] if len(text_content) > 500 else [],
                    confidence=0.7,
                )

        except Exception as e:
            logger.warning(f"Failed to parse Anthropic response: {e}")
            raise VisionProviderError("anthropic", f"Response parsing failed: {str(e)}")

    def _extract_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from response content."""
        import re

        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code block
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
        return "anthropic"

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


def create_anthropic_provider(
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
    **kwargs: Any,
) -> AnthropicVisionProvider:
    """
    Factory function to create Anthropic Vision provider.

    Args:
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        model: Model name (claude-sonnet-4-20250514, claude-3-5-sonnet-20241022, etc.)
        **kwargs: Additional configuration options

    Returns:
        Configured AnthropicVisionProvider instance

    Example:
        >>> provider = create_anthropic_provider(model="claude-sonnet-4-20250514")
        >>> result = await provider.analyze_image(image_bytes)
    """
    return AnthropicVisionProvider(api_key=api_key, model=model, **kwargs)
