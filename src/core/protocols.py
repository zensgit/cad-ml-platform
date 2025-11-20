"""Protocol interfaces to decouple providers from managers.
Initial lightweight definitions; expand as real providers are integrated.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel


class VisionDescription(BaseModel):  # minimal reuse if needed externally
    summary: str
    details: list[str]
    confidence: float


class VisionProviderProtocol(Protocol):
    @property
    def provider_name(self) -> str:
        ...

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True
    ) -> VisionDescription:
        ...


class OcrResult(BaseModel):  # minimal placeholder; actual full model resides elsewhere
    dimensions: list[dict[str, Any]]
    symbols: list[dict[str, Any]]
    title_block: dict[str, Any]
    confidence: float | None = None


class OcrProviderProtocol(Protocol):
    @property
    def name(self) -> str:
        ...

    async def extract(self, image_bytes: bytes, trace_id: str | None = None) -> OcrResult:
        ...
