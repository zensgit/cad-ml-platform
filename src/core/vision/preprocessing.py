"""Image preprocessing and validation for vision analysis.

Provides:
- Image format validation
- Size and dimension checks
- Automatic resizing and optimization
- Format conversion
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .base import VisionDescription, VisionInputError, VisionProvider

logger = logging.getLogger(__name__)


class ImageFormat(Enum):
    """Supported image formats."""

    PNG = "png"
    JPEG = "jpeg"
    GIF = "gif"
    WEBP = "webp"
    BMP = "bmp"
    TIFF = "tiff"
    UNKNOWN = "unknown"


@dataclass
class ImageInfo:
    """Information about an image."""

    format: ImageFormat
    width: int
    height: int
    size_bytes: int
    has_alpha: bool
    color_mode: str
    dpi: Optional[Tuple[int, int]] = None

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        if self.height == 0:
            return 0.0
        return self.width / self.height

    @property
    def megapixels(self) -> float:
        """Calculate megapixels."""
        return (self.width * self.height) / 1_000_000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "format": self.format.value,
            "width": self.width,
            "height": self.height,
            "size_bytes": self.size_bytes,
            "size_kb": self.size_bytes / 1024,
            "has_alpha": self.has_alpha,
            "color_mode": self.color_mode,
            "aspect_ratio": self.aspect_ratio,
            "megapixels": self.megapixels,
            "dpi": self.dpi,
        }


@dataclass
class ValidationResult:
    """Result of image validation."""

    valid: bool
    info: Optional[ImageInfo] = None
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing."""

    # Size limits
    max_size_bytes: int = 20 * 1024 * 1024  # 20MB
    min_size_bytes: int = 100  # 100 bytes minimum
    max_width: int = 4096
    max_height: int = 4096
    min_width: int = 10
    min_height: int = 10

    # Format settings
    allowed_formats: List[ImageFormat] = None
    target_format: Optional[ImageFormat] = None  # Convert to this format
    strip_metadata: bool = False

    # Optimization
    auto_resize: bool = True
    resize_quality: int = 95  # JPEG quality
    max_megapixels: float = 16.0

    # Validation
    validate_content: bool = True
    require_engineering_drawing: bool = False

    def __post_init__(self) -> None:
        if self.allowed_formats is None:
            self.allowed_formats = [
                ImageFormat.PNG,
                ImageFormat.JPEG,
                ImageFormat.GIF,
                ImageFormat.WEBP,
            ]


class ImageValidator:
    """
    Validates images for vision analysis.

    Features:
    - Format detection
    - Size validation
    - Dimension checks
    - Content validation
    """

    # Magic bytes for format detection
    FORMAT_SIGNATURES = {
        b"\x89PNG\r\n\x1a\n": ImageFormat.PNG,
        b"\xff\xd8\xff": ImageFormat.JPEG,
        b"GIF87a": ImageFormat.GIF,
        b"GIF89a": ImageFormat.GIF,
        b"RIFF": ImageFormat.WEBP,  # RIFF....WEBP
        b"BM": ImageFormat.BMP,
        b"II*\x00": ImageFormat.TIFF,  # Little endian
        b"MM\x00*": ImageFormat.TIFF,  # Big endian
    }

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize validator.

        Args:
            config: Preprocessing configuration
        """
        self._config = config or PreprocessingConfig()

    def detect_format(self, image_data: bytes) -> ImageFormat:
        """
        Detect image format from magic bytes.

        Args:
            image_data: Raw image bytes

        Returns:
            Detected ImageFormat
        """
        if len(image_data) < 12:
            return ImageFormat.UNKNOWN

        for signature, format in self.FORMAT_SIGNATURES.items():
            if image_data.startswith(signature):
                # Special check for WebP
                if format == ImageFormat.WEBP:
                    if b"WEBP" in image_data[8:12]:
                        return ImageFormat.WEBP
                    continue
                return format

        return ImageFormat.UNKNOWN

    def get_image_info(self, image_data: bytes) -> ImageInfo:
        """
        Extract detailed information from image.

        Args:
            image_data: Raw image bytes

        Returns:
            ImageInfo with dimensions and metadata
        """
        format = self.detect_format(image_data)
        width, height = self._get_dimensions(image_data, format)

        # Determine color mode and alpha
        has_alpha = False
        color_mode = "RGB"

        if format == ImageFormat.PNG:
            has_alpha = self._png_has_alpha(image_data)
            color_mode = "RGBA" if has_alpha else "RGB"
        elif format == ImageFormat.GIF:
            has_alpha = True  # GIF supports transparency
            color_mode = "P"  # Palette mode
        elif format == ImageFormat.WEBP:
            has_alpha = self._webp_has_alpha(image_data)
            color_mode = "RGBA" if has_alpha else "RGB"

        return ImageInfo(
            format=format,
            width=width,
            height=height,
            size_bytes=len(image_data),
            has_alpha=has_alpha,
            color_mode=color_mode,
        )

    def _get_dimensions(
        self, image_data: bytes, format: ImageFormat
    ) -> Tuple[int, int]:
        """Extract width and height from image data."""
        try:
            if format == ImageFormat.PNG:
                # PNG: width at bytes 16-19, height at 20-23
                if len(image_data) >= 24:
                    width = int.from_bytes(image_data[16:20], "big")
                    height = int.from_bytes(image_data[20:24], "big")
                    return width, height

            elif format == ImageFormat.JPEG:
                # JPEG: need to parse markers
                return self._parse_jpeg_dimensions(image_data)

            elif format == ImageFormat.GIF:
                # GIF: width at bytes 6-7, height at 8-9
                if len(image_data) >= 10:
                    width = int.from_bytes(image_data[6:8], "little")
                    height = int.from_bytes(image_data[8:10], "little")
                    return width, height

            elif format == ImageFormat.WEBP:
                return self._parse_webp_dimensions(image_data)

            elif format == ImageFormat.BMP:
                # BMP: width at 18-21, height at 22-25
                if len(image_data) >= 26:
                    width = int.from_bytes(image_data[18:22], "little")
                    height = abs(int.from_bytes(image_data[22:26], "little", signed=True))
                    return width, height

        except Exception as e:
            logger.warning(f"Failed to parse dimensions: {e}")

        return 0, 0

    def _parse_jpeg_dimensions(self, data: bytes) -> Tuple[int, int]:
        """Parse JPEG dimensions from SOF marker."""
        i = 2  # Skip SOI marker
        while i < len(data) - 9:
            if data[i] != 0xFF:
                break

            marker = data[i + 1]

            # SOF markers (Start Of Frame)
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                height = int.from_bytes(data[i + 5:i + 7], "big")
                width = int.from_bytes(data[i + 7:i + 9], "big")
                return width, height

            # Get segment length and skip
            if marker in (0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7,
                          0xD8, 0xD9, 0x01):
                i += 2
            else:
                length = int.from_bytes(data[i + 2:i + 4], "big")
                i += 2 + length

        return 0, 0

    def _parse_webp_dimensions(self, data: bytes) -> Tuple[int, int]:
        """Parse WebP dimensions."""
        if len(data) < 30:
            return 0, 0

        # Check for VP8 chunk
        if data[12:16] == b"VP8 ":
            # Lossy WebP
            if len(data) >= 30:
                width = int.from_bytes(data[26:28], "little") & 0x3FFF
                height = int.from_bytes(data[28:30], "little") & 0x3FFF
                return width, height

        elif data[12:16] == b"VP8L":
            # Lossless WebP
            if len(data) >= 25:
                signature = int.from_bytes(data[21:25], "little")
                width = (signature & 0x3FFF) + 1
                height = ((signature >> 14) & 0x3FFF) + 1
                return width, height

        elif data[12:16] == b"VP8X":
            # Extended WebP
            if len(data) >= 30:
                width = int.from_bytes(data[24:27], "little") + 1
                height = int.from_bytes(data[27:30], "little") + 1
                return width, height

        return 0, 0

    def _png_has_alpha(self, data: bytes) -> bool:
        """Check if PNG has alpha channel."""
        if len(data) < 26:
            return False
        # Color type at byte 25: 4 = grayscale+alpha, 6 = RGBA
        color_type = data[25]
        return color_type in (4, 6)

    def _webp_has_alpha(self, data: bytes) -> bool:
        """Check if WebP has alpha channel."""
        if len(data) < 21:
            return False
        # VP8X extended format has alpha flag
        if data[12:16] == b"VP8X" and len(data) >= 21:
            flags = data[20]
            return bool(flags & 0x10)
        # VP8L (lossless) may have alpha
        if data[12:16] == b"VP8L":
            return True
        return False

    def validate(self, image_data: bytes) -> ValidationResult:
        """
        Validate image against configuration.

        Args:
            image_data: Raw image bytes

        Returns:
            ValidationResult with validation status
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Check minimum size
        if len(image_data) < self._config.min_size_bytes:
            errors.append(
                f"Image too small: {len(image_data)} bytes "
                f"(minimum: {self._config.min_size_bytes})"
            )
            return ValidationResult(valid=False, errors=errors)

        # Check maximum size
        if len(image_data) > self._config.max_size_bytes:
            errors.append(
                f"Image too large: {len(image_data)} bytes "
                f"(maximum: {self._config.max_size_bytes})"
            )
            return ValidationResult(valid=False, errors=errors)

        # Get image info
        info = self.get_image_info(image_data)

        # Check format
        if info.format == ImageFormat.UNKNOWN:
            errors.append("Unknown or unsupported image format")
            return ValidationResult(valid=False, info=info, errors=errors)

        if info.format not in self._config.allowed_formats:
            errors.append(
                f"Format {info.format.value} not allowed. "
                f"Allowed: {[f.value for f in self._config.allowed_formats]}"
            )
            return ValidationResult(valid=False, info=info, errors=errors)

        # Check dimensions
        if info.width == 0 or info.height == 0:
            errors.append("Could not determine image dimensions")
            return ValidationResult(valid=False, info=info, errors=errors)

        if info.width < self._config.min_width:
            errors.append(
                f"Image width {info.width}px below minimum {self._config.min_width}px"
            )

        if info.height < self._config.min_height:
            errors.append(
                f"Image height {info.height}px below minimum {self._config.min_height}px"
            )

        if info.width > self._config.max_width:
            if self._config.auto_resize:
                warnings.append(
                    f"Image width {info.width}px exceeds maximum {self._config.max_width}px "
                    f"- will be resized"
                )
            else:
                errors.append(
                    f"Image width {info.width}px exceeds maximum {self._config.max_width}px"
                )

        if info.height > self._config.max_height:
            if self._config.auto_resize:
                warnings.append(
                    f"Image height {info.height}px exceeds maximum {self._config.max_height}px "
                    f"- will be resized"
                )
            else:
                errors.append(
                    f"Image height {info.height}px exceeds maximum {self._config.max_height}px"
                )

        # Check megapixels
        if info.megapixels > self._config.max_megapixels:
            if self._config.auto_resize:
                warnings.append(
                    f"Image {info.megapixels:.1f}MP exceeds maximum "
                    f"{self._config.max_megapixels}MP - will be resized"
                )
            else:
                errors.append(
                    f"Image {info.megapixels:.1f}MP exceeds maximum "
                    f"{self._config.max_megapixels}MP"
                )

        return ValidationResult(
            valid=len(errors) == 0,
            info=info,
            errors=errors,
            warnings=warnings,
        )


class ImagePreprocessor:
    """
    Preprocesses images for optimal vision analysis.

    Features:
    - Automatic resizing
    - Format conversion
    - Optimization
    - Metadata stripping

    Note: Requires Pillow for full functionality.
    Falls back to passthrough if Pillow not available.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self._config = config or PreprocessingConfig()
        self._validator = ImageValidator(config)
        self._pillow_available = self._check_pillow()

    @staticmethod
    def _check_pillow() -> bool:
        """Check if Pillow is available."""
        try:
            import importlib.util
            return importlib.util.find_spec("PIL") is not None
        except Exception:
            logger.warning("Pillow not available - preprocessing limited")
            return False

    def preprocess(self, image_data: bytes) -> Tuple[bytes, ImageInfo]:
        """
        Preprocess image for vision analysis.

        Args:
            image_data: Raw image bytes

        Returns:
            Tuple of (processed image bytes, ImageInfo)

        Raises:
            VisionInputError: If image validation fails
        """
        # Validate first
        validation = self._validator.validate(image_data)
        if not validation.valid:
            raise VisionInputError(
                f"Image validation failed: {'; '.join(validation.errors)}"
            )

        info = validation.info

        # If Pillow not available, return original
        if not self._pillow_available:
            return image_data, info

        # Check if preprocessing needed
        needs_processing = (
            (info.width > self._config.max_width) or
            (info.height > self._config.max_height) or
            (info.megapixels > self._config.max_megapixels) or
            (self._config.target_format and
             info.format != self._config.target_format) or
            self._config.strip_metadata
        )

        if not needs_processing:
            return image_data, info

        # Process with Pillow
        return self._process_with_pillow(image_data, info)

    def _process_with_pillow(
        self, image_data: bytes, info: ImageInfo
    ) -> Tuple[bytes, ImageInfo]:
        """Process image using Pillow."""
        from PIL import Image

        # Open image
        img = Image.open(io.BytesIO(image_data))

        # Resize if needed
        if (info.width > self._config.max_width or
                info.height > self._config.max_height or
                info.megapixels > self._config.max_megapixels):
            img = self._resize_image(img)

        # Convert format if needed
        target_format = self._config.target_format or info.format

        # Handle alpha channel for JPEG
        if target_format == ImageFormat.JPEG and img.mode in ("RGBA", "P"):
            # Convert to RGB, removing alpha
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[3])
            img = background

        # Save to buffer
        buffer = io.BytesIO()

        if target_format == ImageFormat.JPEG:
            img.save(
                buffer,
                format="JPEG",
                quality=self._config.resize_quality,
                optimize=True,
            )
        elif target_format == ImageFormat.PNG:
            img.save(buffer, format="PNG", optimize=True)
        elif target_format == ImageFormat.WEBP:
            img.save(
                buffer,
                format="WEBP",
                quality=self._config.resize_quality,
            )
        else:
            # Default to PNG
            img.save(buffer, format="PNG")

        processed_data = buffer.getvalue()

        # Update info
        new_info = self._validator.get_image_info(processed_data)
        return processed_data, new_info

    def _resize_image(self, img: Any) -> Any:
        """Resize image to fit within limits."""
        from PIL import Image  # noqa: F401

        width, height = img.size

        # Calculate scale factor
        scale = 1.0

        if width > self._config.max_width:
            scale = min(scale, self._config.max_width / width)

        if height > self._config.max_height:
            scale = min(scale, self._config.max_height / height)

        # Check megapixels
        current_mp = (width * height) / 1_000_000
        if current_mp > self._config.max_megapixels:
            mp_scale = (self._config.max_megapixels / current_mp) ** 0.5
            scale = min(scale, mp_scale)

        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS,
            )
            logger.debug(
                f"Resized image from {width}x{height} to {new_width}x{new_height}"
            )

        return img


class PreprocessingVisionProvider:
    """
    Wrapper that adds preprocessing to any VisionProvider.

    Automatically validates and preprocesses images before analysis.
    """

    def __init__(
        self,
        provider: VisionProvider,
        config: Optional[PreprocessingConfig] = None,
    ):
        """
        Initialize preprocessing provider.

        Args:
            provider: The underlying vision provider
            config: Preprocessing configuration
        """
        self._provider = provider
        self._preprocessor = ImagePreprocessor(config)
        self._validator = ImageValidator(config)

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
        skip_preprocessing: bool = False,
    ) -> VisionDescription:
        """
        Preprocess and analyze image.

        Args:
            image_data: Raw image bytes
            include_description: Whether to generate description
            skip_preprocessing: Skip preprocessing (just validate)

        Returns:
            VisionDescription with analysis results

        Raises:
            VisionInputError: If image validation fails
        """
        if skip_preprocessing:
            # Just validate
            validation = self._validator.validate(image_data)
            if not validation.valid:
                raise VisionInputError(
                    f"Image validation failed: {'; '.join(validation.errors)}"
                )
            processed_data = image_data
        else:
            # Preprocess
            processed_data, info = self._preprocessor.preprocess(image_data)
            logger.debug(
                f"Preprocessed image: {info.format.value} "
                f"{info.width}x{info.height} ({info.size_bytes} bytes)"
            )

        return await self._provider.analyze_image(
            processed_data, include_description
        )

    def validate(self, image_data: bytes) -> ValidationResult:
        """
        Validate image without processing.

        Args:
            image_data: Raw image bytes

        Returns:
            ValidationResult with validation status
        """
        return self._validator.validate(image_data)

    def get_image_info(self, image_data: bytes) -> ImageInfo:
        """
        Get image information without processing.

        Args:
            image_data: Raw image bytes

        Returns:
            ImageInfo with image details
        """
        return self._validator.get_image_info(image_data)

    @property
    def provider_name(self) -> str:
        """Return wrapped provider name."""
        return self._provider.provider_name

    @property
    def preprocessor(self) -> ImagePreprocessor:
        """Get the preprocessor."""
        return self._preprocessor


def create_preprocessing_provider(
    provider: VisionProvider,
    max_size_mb: float = 20.0,
    max_dimension: int = 4096,
    auto_resize: bool = True,
    target_format: Optional[ImageFormat] = None,
) -> PreprocessingVisionProvider:
    """
    Factory to create a preprocessing provider wrapper.

    Args:
        provider: The underlying vision provider
        max_size_mb: Maximum image size in MB
        max_dimension: Maximum width/height
        auto_resize: Auto-resize oversized images
        target_format: Convert to this format

    Returns:
        PreprocessingVisionProvider wrapping the original

    Example:
        >>> provider = create_vision_provider("openai")
        >>> preprocess = create_preprocessing_provider(
        ...     provider,
        ...     max_size_mb=10.0,
        ...     max_dimension=2048,
        ... )
        >>> result = await preprocess.analyze_image(large_image_bytes)
    """
    config = PreprocessingConfig(
        max_size_bytes=int(max_size_mb * 1024 * 1024),
        max_width=max_dimension,
        max_height=max_dimension,
        auto_resize=auto_resize,
        target_format=target_format,
    )
    return PreprocessingVisionProvider(provider=provider, config=config)
