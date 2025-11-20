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

import src.core.config as config
from src.utils.metrics import (
    update_vision_error_ema,
    vision_errors_total,
    vision_image_size_bytes,
    vision_input_rejected_total,
    vision_processing_duration_seconds,
    vision_requests_total,
)
from src.utils.metrics_helpers import safe_inc, safe_observe

from .base import (
    OcrResult,
    VisionAnalyzeRequest,
    VisionAnalyzeResponse,
    VisionDescription,
    VisionInputError,
    VisionProvider,
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
        ocr_manager: Optional[object] = None,  # Type: OcrManager from src.core.ocr.manager
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
        provider_name = self.vision_provider.provider_name
        safe_inc(vision_requests_total, provider=provider_name, status="start")

        # Load image first; propagate input errors to caller
        try:
            image_bytes = await self._load_image(request)
        except VisionInputError:
            safe_inc(vision_requests_total, provider=provider_name, status="input_error")
            safe_inc(vision_errors_total, provider=provider_name, code="input_error")
            update_vision_error_ema(True)
            raise
        except Exception:
            # Propagate unexpected errors; API layer will standardize response
            try:
                processing_time_ms = (time.time() - start_time) * 1000
                safe_inc(vision_requests_total, provider=provider_name, status="error")
                safe_observe(
                    vision_processing_duration_seconds,
                    processing_time_ms / 1000.0,
                    provider=provider_name,
                )
                safe_inc(vision_errors_total, provider=provider_name, code="internal")
                update_vision_error_ema(True)
            except Exception:
                pass
            raise

        # Record input size histogram
        try:
            vision_image_size_bytes.observe(len(image_bytes))
        except Exception:
            pass

        # Process vision + optional OCR
        try:
            description: Optional[VisionDescription] = None
            if request.include_description:
                description = await self.vision_provider.analyze_image(
                    image_data=image_bytes, include_description=True
                )
            ocr_result: Optional[OcrResult] = None
            if request.include_ocr and self.ocr_manager:
                ocr_result = await self._extract_ocr(
                    image_bytes=image_bytes, provider=request.ocr_provider
                )
            processing_time_ms = (time.time() - start_time) * 1000
            safe_inc(vision_requests_total, provider=provider_name, status="success")
            safe_observe(
                vision_processing_duration_seconds,
                processing_time_ms / 1000.0,
                provider=provider_name,
            )
            update_vision_error_ema(False)
            return VisionAnalyzeResponse(
                success=True,
                description=description,
                ocr=ocr_result,
                provider=provider_name,
                processing_time_ms=processing_time_ms,
            )
        except Exception:
            # Propagate unexpected errors; API layer will standardize response
            try:
                processing_time_ms = (time.time() - start_time) * 1000
                safe_inc(vision_requests_total, provider=provider_name, status="error")
                safe_observe(
                    vision_processing_duration_seconds,
                    processing_time_ms / 1000.0,
                    provider=provider_name,
                )
                safe_inc(vision_errors_total, provider=provider_name, code="internal")
                update_vision_error_ema(True)
            except Exception:
                pass
            raise

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
        settings = config.get_settings()
        max_bytes = settings.VISION_MAX_BASE64_BYTES
        if request.image_base64:
            # Decode base64 image with strict validation
            import base64

            try:
                # Use validate=True for stricter base64 validation
                decoded = base64.b64decode(request.image_base64, validate=True)
                if len(decoded) > max_bytes:
                    size_mb = len(decoded) / 1024 / 1024
                    limit_mb = max_bytes / 1024 / 1024
                    vision_input_rejected_total.labels(reason="base64_too_large").inc()
                    raise VisionInputError(
                        f"Image too large ({size_mb:.2f}MB) via base64. Max {limit_mb:.2f}MB."
                    )
                if len(decoded) == 0:
                    vision_input_rejected_total.labels(reason="base64_empty").inc()
                    raise VisionInputError("Decoded image is empty (0 bytes)")
                return decoded
            except Exception as e:
                # If we intentionally raised VisionInputError above, re-raise without reclassification
                if isinstance(e, VisionInputError):
                    raise
                # Try to classify base64 error subtype
                msg = str(e).lower()
                if "incorrect padding" in msg:
                    vision_input_rejected_total.labels(reason="base64_padding_error").inc()
                elif "non-base64" in msg or "invalid" in msg:
                    vision_input_rejected_total.labels(reason="base64_invalid_char").inc()
                else:
                    vision_input_rejected_total.labels(reason="base64_decode_error").inc()
                raise VisionInputError(f"Invalid base64 image data: {e}")

        elif request.image_url:
            # Download from URL with validation and size limits
            from urllib.parse import urlparse

            import httpx

            # Validate URL scheme (only http/https)
            try:
                parsed_url = urlparse(request.image_url)
                if parsed_url.scheme not in ["http", "https"]:
                    vision_input_rejected_total.labels(reason="url_invalid_scheme").inc()
                    raise VisionInputError(
                        f"Invalid URL scheme '{parsed_url.scheme}'. Only http/https supported."
                    )
            except Exception as e:
                vision_input_rejected_total.labels(reason="url_invalid_format").inc()
                raise VisionInputError(f"Invalid URL format: {e}")

            # Download image with timeout and size limit
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(request.image_url, follow_redirects=True)

                    # Check HTTP status
                    if response.status_code == 404:
                        vision_input_rejected_total.labels(reason="url_not_found").inc()
                        raise VisionInputError(
                            f"Image not found at URL (HTTP 404): {request.image_url}"
                        )
                    elif response.status_code == 403:
                        vision_input_rejected_total.labels(reason="url_forbidden").inc()
                        raise VisionInputError(
                            f"Access forbidden to URL (HTTP 403): {request.image_url}"
                        )
                    elif response.status_code >= 400:
                        code = response.status_code
                        vision_input_rejected_total.labels(reason="url_http_error").inc()
                        raise VisionInputError(
                            f"Failed to download image (HTTP {code}): {request.image_url}"
                        )

                    # Check content length (50MB limit)
                    content_length = response.headers.get("content-length")
                    max_size_bytes = 50 * 1024 * 1024  # 50MB

                    if content_length and int(content_length) > max_size_bytes:
                        size_mb = int(content_length) / 1024 / 1024
                        vision_input_rejected_total.labels(reason="url_too_large_header").inc()
                        raise VisionInputError(f"Image too large ({size_mb:.1f}MB). Max 50MB.")

                    # Read content and check actual size
                    image_data = response.content
                    if len(image_data) > max_size_bytes:
                        size_mb = len(image_data) / 1024 / 1024
                        vision_input_rejected_total.labels(reason="url_too_large_download").inc()
                        raise VisionInputError(
                            f"Downloaded image too large ({size_mb:.1f}MB). Max 50MB."
                        )

                    if len(image_data) == 0:
                        vision_input_rejected_total.labels(reason="url_empty").inc()
                        raise VisionInputError("Downloaded image is empty (0 bytes)")

                    return image_data

            except httpx.TimeoutException:
                vision_input_rejected_total.labels(reason="url_timeout").inc()
                raise VisionInputError(
                    f"Timeout downloading image from URL (>5s): {request.image_url}"
                )
            except httpx.RequestError as e:
                vision_input_rejected_total.labels(reason="url_network_error").inc()
                raise VisionInputError(f"Network error downloading image: {e}")
            except VisionInputError:
                # Re-raise VisionInputError as-is
                raise
            except Exception as e:
                vision_input_rejected_total.labels(reason="url_download_error").inc()
                raise VisionInputError(f"Failed to download image from URL: {e}")

        else:
            raise VisionInputError("Either image_url or image_base64 must be provided")

    async def _extract_ocr(self, image_bytes: bytes, provider: str = "auto") -> Optional[OcrResult]:
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
                image_bytes=image_bytes, strategy=provider
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
                fallback_level=getattr(ocr_raw_result, "fallback_level", None),
                confidence=ocr_raw_result.calibrated_confidence or ocr_raw_result.confidence,
            )

        except Exception as e:
            # Log error but don't fail entire request (graceful degradation)
            # Vision description will still be returned even if OCR fails
            # TODO: Add proper structured logging
            import logging

            logging.getLogger(__name__).warning(
                "vision.ocr_extract_failed",
                extra={"error": str(e)},
            )
            return None
