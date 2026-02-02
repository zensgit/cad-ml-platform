"""
DWG file parser for basic header and entity information.

Provides limited direct DWG parsing capabilities.
For full support, use DWGConverter with ODA File Converter.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class DWGVersion(str, Enum):
    """DWG file versions."""
    R13 = "AC1012"
    R14 = "AC1014"
    R2000 = "AC1015"
    R2004 = "AC1018"
    R2007 = "AC1021"
    R2010 = "AC1024"
    R2013 = "AC1027"
    R2018 = "AC1032"
    UNKNOWN = "UNKNOWN"


@dataclass
class DWGHeader:
    """DWG file header information."""
    version: DWGVersion = DWGVersion.UNKNOWN
    version_string: str = ""
    file_size: int = 0
    codepage: int = 0
    creation_date: Optional[float] = None
    modification_date: Optional[float] = None
    thumbnail_offset: int = 0
    is_encrypted: bool = False

    # Drawing settings (if parsed)
    drawing_units: int = 0
    drawing_base: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    extents_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    extents_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version.value,
            "version_string": self.version_string,
            "file_size": self.file_size,
            "codepage": self.codepage,
            "creation_date": self.creation_date,
            "modification_date": self.modification_date,
            "is_encrypted": self.is_encrypted,
            "drawing_units": self.drawing_units,
            "extents_min": self.extents_min,
            "extents_max": self.extents_max,
        }


@dataclass
class DWGEntity:
    """Basic DWG entity information."""
    handle: int = 0
    entity_type: str = ""
    layer: str = "0"
    color: int = 256  # ByLayer
    linetype: str = "ByLayer"

    # Geometry (simplified)
    points: List[Tuple[float, float, float]] = field(default_factory=list)

    # Additional properties
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "handle": self.handle,
            "entity_type": self.entity_type,
            "layer": self.layer,
            "color": self.color,
            "linetype": self.linetype,
            "points": self.points,
            "properties": self.properties,
        }


class DWGParser:
    """
    Basic DWG file parser.

    Provides limited parsing of DWG header and structure.
    For full entity parsing, convert to DXF using DWGConverter.

    Note: DWG is a proprietary binary format. This parser
    provides basic header information only.
    """

    # DWG magic bytes for version detection
    _VERSION_MAP = {
        b"AC1012": DWGVersion.R13,
        b"AC1014": DWGVersion.R14,
        b"AC1015": DWGVersion.R2000,
        b"AC1018": DWGVersion.R2004,
        b"AC1021": DWGVersion.R2007,
        b"AC1024": DWGVersion.R2010,
        b"AC1027": DWGVersion.R2013,
        b"AC1032": DWGVersion.R2018,
    }

    def __init__(self, file_path: Optional[Union[str, Path]] = None):
        """
        Initialize DWG parser.

        Args:
            file_path: Path to DWG file (optional)
        """
        self._file_path: Optional[Path] = None
        self._header: Optional[DWGHeader] = None
        self._entities: List[DWGEntity] = []
        self._raw_data: Optional[bytes] = None

        if file_path:
            self.load(file_path)

    def load(self, file_path: Union[str, Path]) -> "DWGParser":
        """
        Load a DWG file.

        Args:
            file_path: Path to DWG file

        Returns:
            Self for chaining
        """
        self._file_path = Path(file_path)

        if not self._file_path.exists():
            raise FileNotFoundError(f"DWG file not found: {self._file_path}")

        if not self._file_path.suffix.lower() == ".dwg":
            raise ValueError(f"Not a DWG file: {self._file_path}")

        with open(self._file_path, "rb") as f:
            self._raw_data = f.read()

        self._parse_header()
        return self

    def load_bytes(self, data: bytes) -> "DWGParser":
        """
        Load DWG from bytes.

        Args:
            data: DWG file content as bytes

        Returns:
            Self for chaining
        """
        self._raw_data = data
        self._file_path = None
        self._parse_header()
        return self

    def _parse_header(self) -> None:
        """Parse DWG file header."""
        if not self._raw_data or len(self._raw_data) < 128:
            raise ValueError("Invalid DWG data: too short")

        self._header = DWGHeader()
        self._header.file_size = len(self._raw_data)

        # Parse version from magic bytes (first 6 bytes)
        version_bytes = self._raw_data[:6]
        self._header.version_string = version_bytes.decode("ascii", errors="replace")
        self._header.version = self._VERSION_MAP.get(version_bytes, DWGVersion.UNKNOWN)

        if self._header.version == DWGVersion.UNKNOWN:
            logger.warning(f"Unknown DWG version: {version_bytes}")

        # Parse additional header fields based on version
        try:
            self._parse_version_specific_header()
        except Exception as e:
            logger.warning(f"Failed to parse extended header: {e}")

    def _parse_version_specific_header(self) -> None:
        """Parse version-specific header fields."""
        if not self._header or not self._raw_data:
            return

        # R2000+ header structure
        if self._header.version in (DWGVersion.R2000, DWGVersion.R2004,
                                     DWGVersion.R2007, DWGVersion.R2010,
                                     DWGVersion.R2013, DWGVersion.R2018):
            try:
                # Byte 6: Maintenance release version
                # Byte 7: Unknown
                # Bytes 8-11: Preview image address
                if len(self._raw_data) >= 12:
                    self._header.thumbnail_offset = struct.unpack("<I", self._raw_data[8:12])[0]

                # Bytes 12-13: Codepage
                if len(self._raw_data) >= 14:
                    self._header.codepage = struct.unpack("<H", self._raw_data[12:14])[0]

                # Check for encryption (simplified)
                # In newer versions, certain bits indicate encryption
                if len(self._raw_data) >= 7:
                    self._header.is_encrypted = (self._raw_data[6] & 0x04) != 0

            except struct.error as e:
                logger.debug(f"Header parsing error: {e}")

    @property
    def header(self) -> Optional[DWGHeader]:
        """Get parsed header."""
        return self._header

    @property
    def version(self) -> DWGVersion:
        """Get DWG version."""
        return self._header.version if self._header else DWGVersion.UNKNOWN

    @property
    def file_path(self) -> Optional[Path]:
        """Get file path."""
        return self._file_path

    @property
    def entities(self) -> List[DWGEntity]:
        """
        Get parsed entities.

        Note: Direct entity parsing is limited.
        Convert to DXF for full entity access.
        """
        return self._entities

    def get_thumbnail(self) -> Optional[bytes]:
        """
        Extract thumbnail image if present.

        Returns:
            Thumbnail image bytes (BMP format) or None
        """
        if not self._header or not self._raw_data:
            return None

        if self._header.thumbnail_offset == 0:
            return None

        try:
            offset = self._header.thumbnail_offset
            if offset >= len(self._raw_data):
                return None

            # Read thumbnail section
            # Format: sentinel + size + BMP data
            # This is simplified - actual format is more complex

            # Skip to BMP data (offset may vary by version)
            bmp_start = offset + 16  # Skip sentinel and headers

            if bmp_start + 14 >= len(self._raw_data):
                return None

            # Check for BMP signature
            if self._raw_data[bmp_start:bmp_start+2] == b"BM":
                # Read BMP size
                bmp_size = struct.unpack("<I", self._raw_data[bmp_start+2:bmp_start+6])[0]
                if bmp_start + bmp_size <= len(self._raw_data):
                    return self._raw_data[bmp_start:bmp_start+bmp_size]

            return None

        except Exception as e:
            logger.debug(f"Thumbnail extraction error: {e}")
            return None

    def is_valid(self) -> bool:
        """Check if file is a valid DWG."""
        if not self._header:
            return False
        return self._header.version != DWGVersion.UNKNOWN

    def get_info(self) -> Dict[str, Any]:
        """Get file information summary."""
        return {
            "file_path": str(self._file_path) if self._file_path else None,
            "is_valid": self.is_valid(),
            "version": self.version.value,
            "header": self._header.to_dict() if self._header else None,
            "entity_count": len(self._entities),
            "has_thumbnail": self._header.thumbnail_offset > 0 if self._header else False,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert parsed data to dictionary."""
        return {
            "header": self._header.to_dict() if self._header else None,
            "entities": [e.to_dict() for e in self._entities],
        }


def parse_dwg_header(file_path: Union[str, Path]) -> DWGHeader:
    """
    Convenience function to parse DWG header.

    Args:
        file_path: Path to DWG file

    Returns:
        DWGHeader
    """
    parser = DWGParser(file_path)
    if parser.header is None:
        raise ValueError(f"Failed to parse DWG header: {file_path}")
    return parser.header


def detect_dwg_version(file_path: Union[str, Path]) -> DWGVersion:
    """
    Detect DWG file version.

    Args:
        file_path: Path to DWG file

    Returns:
        DWGVersion
    """
    parser = DWGParser(file_path)
    return parser.version


def is_dwg_file(file_path: Union[str, Path]) -> bool:
    """
    Check if file is a valid DWG.

    Args:
        file_path: Path to file

    Returns:
        True if valid DWG file
    """
    try:
        parser = DWGParser(file_path)
        return parser.is_valid()
    except Exception:
        return False
