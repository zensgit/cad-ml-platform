"""DeepSeek Vision Stub Provider.

Placeholder implementation for DeepSeek-VL that returns fixed responses.
Used for:
- MVP development and testing
- Integration testing without GPU dependency
- Architecture validation

Future replacement: Real DeepSeek-VL model with transformers/vLLM.
"""

from __future__ import annotations

import asyncio
from typing import Optional  # noqa: F401 (kept for future optional params)

from ..base import VisionDescription, VisionProvider


class DeepSeekStubProvider(VisionProvider):
    """
    Stub provider returning fixed vision descriptions.

    Simulates DeepSeek-VL behavior for testing and MVP development.
    """

    def __init__(self, simulate_latency_ms: float = 50.0):
        """
        Initialize stub provider.

        Args:
            simulate_latency_ms: Simulated processing latency in milliseconds
        """
        self.simulate_latency_ms = simulate_latency_ms

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True
    ) -> VisionDescription:
        """
        Return fixed vision description.

        Args:
            image_data: Raw image bytes (unused in stub, but validated)
            include_description: Whether to include description

        Returns:
            Fixed VisionDescription

        Note:
            This stub always returns the same description regardless of input.
            Real implementation will process image_data with DeepSeek-VL model.
        """
        # Simulate processing latency
        if self.simulate_latency_ms > 0:
            await asyncio.sleep(self.simulate_latency_ms / 1000.0)

        # Validate input
        if not image_data or len(image_data) == 0:
            raise ValueError("image_data cannot be empty")

        # Return fixed description (stub behavior)
        if not include_description:
            # Minimal description for OCR-only mode
            return VisionDescription(
                summary="Image processed (OCR-only mode)", details=[], confidence=1.0
            )

        # Full description (typical case)
        return VisionDescription(
            summary="This is a mechanical engineering drawing showing a cylindrical part with threaded features.",
            details=[
                "Main body features a diameter dimension of approximately 20mm with bilateral tolerance",
                "External thread specification visible (M10×1.5 pitch)",
                "Surface finish requirement indicated (Ra 3.2 or similar)",
                "Title block present with drawing number and material specification",
                "Standard orthographic projection with front and side views",
            ],
            confidence=0.92,
        )

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "deepseek_stub"


# ========== Factory Function ==========


def create_stub_provider(simulate_latency_ms: float = 50.0) -> DeepSeekStubProvider:
    """
    Factory function to create stub provider.

    Args:
        simulate_latency_ms: Simulated latency (default: 50ms)

    Returns:
        Configured DeepSeekStubProvider instance

    Example:
        >>> provider = create_stub_provider(simulate_latency_ms=100)
        >>> result = await provider.analyze_image(image_bytes)
    """
    return DeepSeekStubProvider(simulate_latency_ms=simulate_latency_ms)
