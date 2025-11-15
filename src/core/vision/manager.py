"""Vision Manager: orchestrates VisionProvider + OCRManager for end-to-end analysis.

Responsibilities:
- Coordinate vision description generation (VisionProvider)
- Coordinate OCR extraction (OCRManager)
- Aggregate results into unified response
- Handle errors and fallbacks
"""

from __future__ import annotations

import time
from typing import Optional

from .base import (
    VisionAnalyzeRequest,
    VisionAnalyzeResponse,
    VisionDescription,
    OcrResult,
    VisionProvider,
    VisionInputError,
)


class VisionManager:
    """
    Orchestrates vision analysis with OCR integration.

    Workflow:
    1. Validate input (image_url or image_base64)
    2. Load image bytes
    3. Vision analysis (VisionProvider)
    4. OCR extraction (OCRManager, if requested)
    5. Aggregate results
    """

    def __init__(
        self,
        vision_provider: VisionProvider,
        ocr_manager: Optional[object] = None  # Type: OcrManager from src.core.ocr.manager
    ):
        """
        Initialize VisionManager.

        Args:
            vision_provider: VisionProvider instance (e.g., DeepSeekStubProvider)
            ocr_manager: Optional OcrManager for dimension/symbol extraction
        """
        self.vision_provider = vision_provider
        self.ocr_manager = ocr_manager

    async def analyze(self, request: VisionAnalyzeRequest) -> VisionAnalyzeResponse:
        """
        Perform end-to-end vision analysis.

        Args:
            request: VisionAnalyzeRequest with image and options

        Returns:
            VisionAnalyzeResponse with description, OCR results, and metadata

        Raises:
            VisionInputError: If both image_url and image_base64 are missing
        """
        start_time = time.time()

        try:
            # Step 1: Load image bytes
            image_bytes = await self._load_image(request)

            # Step 2: Vision description (if requested)
            description: Optional[VisionDescription] = None
            if request.include_description:
                description = await self.vision_provider.analyze_image(
                    image_data=image_bytes,
                    include_description=True
                )

            # Step 3: OCR extraction (if requested and manager available)
            ocr_result: Optional[OcrResult] = None
            if request.include_ocr and self.ocr_manager:
                ocr_result = await self._extract_ocr(
                    image_bytes=image_bytes,
                    provider=request.ocr_provider
                )

            # Step 4: Aggregate results
            processing_time_ms = (time.time() - start_time) * 1000

            return VisionAnalyzeResponse(
                success=True,
                description=description,
                ocr=ocr_result,
                provider=self.vision_provider.provider_name,
                processing_time_ms=processing_time_ms
            )

        except VisionInputError:
            # Re-raise input validation errors so they can be handled as 400 at API level
            raise

        except Exception as e:
            # Other errors: return error response with success=False
            processing_time_ms = (time.time() - start_time) * 1000

            return VisionAnalyzeResponse(
                success=False,
                description=None,
                ocr=None,
                provider=self.vision_provider.provider_name,
                processing_time_ms=processing_time_ms,
                error=str(e)
            )

    async def _load_image(self, request: VisionAnalyzeRequest) -> bytes:
        """
        Load image bytes from URL or base64.

        Args:
            request: VisionAnalyzeRequest

        Returns:
            Raw image bytes

        Raises:
            VisionInputError: If both image_url and image_base64 are missing
        """
        if request.image_base64:
            # Decode base64 image with strict validation
            import base64
            try:
                # Use validate=True for stricter base64 validation
                return base64.b64decode(request.image_base64, validate=True)
            except Exception as e:
                raise VisionInputError(f"Invalid base64 image data: {e}")

        elif request.image_url:
            # Download from URL with validation and size limits
            import httpx
            from urllib.parse import urlparse

            # Validate URL scheme (only http/https)
            try:
                parsed_url = urlparse(request.image_url)
                if parsed_url.scheme not in ['http', 'https']:
                    raise VisionInputError(
                        f"Invalid URL scheme '{parsed_url.scheme}'. Only http:// and https:// are supported."
                    )
            except Exception as e:
                raise VisionInputError(f"Invalid URL format: {e}")

            # Download image with timeout and size limit
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(request.image_url, follow_redirects=True)

                    # Check HTTP status
                    if response.status_code == 404:
                        raise VisionInputError(f"Image not found at URL (HTTP 404): {request.image_url}")
                    elif response.status_code == 403:
                        raise VisionInputError(f"Access forbidden to URL (HTTP 403): {request.image_url}")
                    elif response.status_code >= 400:
                        raise VisionInputError(f"Failed to download image (HTTP {response.status_code}): {request.image_url}")

                    # Check content length (50MB limit)
                    content_length = response.headers.get('content-length')
                    max_size_bytes = 50 * 1024 * 1024  # 50MB

                    if content_length and int(content_length) > max_size_bytes:
                        raise VisionInputError(
                            f"Image too large ({int(content_length) / 1024 / 1024:.1f}MB). Maximum size is 50MB."
                        )

                    # Read content and check actual size
                    image_data = response.content
                    if len(image_data) > max_size_bytes:
                        raise VisionInputError(
                            f"Downloaded image too large ({len(image_data) / 1024 / 1024:.1f}MB). Maximum size is 50MB."
                        )

                    if len(image_data) == 0:
                        raise VisionInputError("Downloaded image is empty (0 bytes)")

                    return image_data

            except httpx.TimeoutException:
                raise VisionInputError(f"Timeout downloading image from URL (>5s): {request.image_url}")
            except httpx.RequestError as e:
                raise VisionInputError(f"Network error downloading image: {e}")
            except VisionInputError:
                # Re-raise VisionInputError as-is
                raise
            except Exception as e:
                raise VisionInputError(f"Failed to download image from URL: {e}")

        else:
            raise VisionInputError(
                "Either image_url or image_base64 must be provided"
            )

    async def _extract_ocr(
        self,
        image_bytes: bytes,
        provider: str = "auto"
    ) -> Optional[OcrResult]:
        """
        Extract OCR data using OCRManager.

        Args:
            image_bytes: Raw image bytes
            provider: OCR provider strategy (auto|paddle|deepseek)

        Returns:
            OcrResult or None if extraction fails

        Note:
            This is a thin wrapper around OCRManager.extract().
            Gracefully degrades: if OCR fails, vision description still returns.
        """
        if not self.ocr_manager:
            return None

        try:
            # Call OCRManager.extract() (from src.core.ocr.manager)
            # This returns src.core.ocr.base.OcrResult
            ocr_raw_result = await self.ocr_manager.extract(
                image_bytes=image_bytes,
                strategy=provider
            )

            # Convert OCR module's OcrResult to Vision module's OcrResult
            # OCR module uses Pydantic models (DimensionInfo, SymbolInfo, TitleBlock)
            # Vision module expects plain dicts for API response
            dimensions_dict = [dim.model_dump() for dim in ocr_raw_result.dimensions]
            symbols_dict = [sym.model_dump() for sym in ocr_raw_result.symbols]
            title_block_dict = ocr_raw_result.title_block.model_dump()

            return OcrResult(
                dimensions=dimensions_dict,
                symbols=symbols_dict,
                title_block=title_block_dict,
                fallback_level=getattr(ocr_raw_result, 'fallback_level', None),
                confidence=ocr_raw_result.calibrated_confidence or ocr_raw_result.confidence
            )

        except Exception as e:
            # Log error but don't fail entire request (graceful degradation)
            # Vision description will still be returned even if OCR fails
            # TODO: Add proper structured logging
            print(f"⚠️ OCR extraction failed (vision description will still be returned): {e}")
            return None
