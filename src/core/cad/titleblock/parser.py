"""
Titleblock Parser - Main interface for titleblock parsing.

Provides:
- Unified titleblock parsing interface
- Integration of detection, templates, and extraction
- Metadata output
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.cad.titleblock.region_detector import (
    RegionDetector,
    TitleblockRegion,
    DetectionMethod,
    BoundingBox,
)
from src.core.cad.titleblock.template_library import (
    TemplateLibrary,
    TitleblockTemplate,
    TemplateMatch,
    FieldType,
)
from src.core.cad.titleblock.field_extractor import (
    FieldExtractor,
    ExtractionResult,
)

logger = logging.getLogger(__name__)


@dataclass
class TitleblockMetadata:
    """Extracted titleblock metadata."""
    # Core fields
    part_number: Optional[str] = None
    drawing_title: Optional[str] = None
    material: Optional[str] = None
    author: Optional[str] = None
    checker: Optional[str] = None
    approver: Optional[str] = None
    date: Optional[str] = None
    revision: Optional[str] = None
    scale: Optional[str] = None
    sheet: Optional[str] = None
    weight: Optional[str] = None
    company: Optional[str] = None
    project: Optional[str] = None

    # Custom fields
    custom_fields: Dict[str, str] = field(default_factory=dict)

    # Extraction metadata
    confidence: float = 0.0
    template_name: Optional[str] = None
    detection_method: Optional[str] = None
    region_bounds: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "part_number": self.part_number,
            "drawing_title": self.drawing_title,
            "material": self.material,
            "author": self.author,
            "checker": self.checker,
            "approver": self.approver,
            "date": self.date,
            "revision": self.revision,
            "scale": self.scale,
            "sheet": self.sheet,
            "weight": self.weight,
            "company": self.company,
            "project": self.project,
            "custom_fields": self.custom_fields,
            "confidence": round(self.confidence, 4),
            "template_name": self.template_name,
            "detection_method": self.detection_method,
            "region_bounds": self.region_bounds,
        }
        # Remove None values for cleaner output
        return {k: v for k, v in result.items() if v is not None}

    @property
    def has_core_fields(self) -> bool:
        """Check if core fields are present."""
        return bool(self.part_number or self.drawing_title)

    @property
    def field_count(self) -> int:
        """Count of non-empty fields."""
        core_fields = [
            self.part_number, self.drawing_title, self.material,
            self.author, self.checker, self.approver, self.date,
            self.revision, self.scale, self.sheet, self.weight,
            self.company, self.project,
        ]
        return sum(1 for f in core_fields if f) + len(self.custom_fields)


@dataclass
class ParserConfig:
    """Configuration for titleblock parser."""
    detection_method: DetectionMethod = DetectionMethod.CORNER_BASED
    auto_detect_template: bool = True
    template_name: Optional[str] = None
    use_ocr: bool = False
    min_confidence: float = 0.3


class TitleblockParser:
    """
    Main titleblock parser.

    Provides:
    - Region detection
    - Template matching
    - Field extraction
    - OCR integration
    """

    def __init__(self, config: Optional[ParserConfig] = None):
        """
        Initialize titleblock parser.

        Args:
            config: Parser configuration
        """
        self._config = config or ParserConfig()
        self._region_detector = RegionDetector()
        self._template_library = TemplateLibrary()
        self._field_extractor = FieldExtractor()

    @property
    def template_library(self) -> TemplateLibrary:
        """Get template library."""
        return self._template_library

    def parse(
        self,
        dxf_path: Path,
        template_name: Optional[str] = None,
    ) -> TitleblockMetadata:
        """
        Parse titleblock from DXF file.

        Args:
            dxf_path: Path to DXF file
            template_name: Optional template to use

        Returns:
            TitleblockMetadata
        """
        try:
            import ezdxf
        except ImportError:
            raise ImportError("ezdxf is required for titleblock parsing")

        # Load DXF file
        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()
        entities = list(msp)

        return self.parse_from_entities(entities, template_name)

    def parse_from_entities(
        self,
        entities: List[Any],
        template_name: Optional[str] = None,
    ) -> TitleblockMetadata:
        """
        Parse titleblock from entity list.

        Args:
            entities: List of DXF entities
            template_name: Optional template to use

        Returns:
            TitleblockMetadata
        """
        # 1. Detect titleblock region
        region = self._region_detector.detect(
            entities,
            method=self._config.detection_method,
        )

        if region is None:
            logger.warning("Could not detect titleblock region")
            return TitleblockMetadata(confidence=0.0)

        # 2. Extract raw text for template matching
        raw_texts = self._extract_text_labels(entities, region.bounds)

        # 3. Match or select template
        template = None
        if template_name or self._config.template_name:
            name = template_name or self._config.template_name
            template = self._template_library.get_template(name)
            if template is None:
                logger.warning(f"Template '{name}' not found, using auto-detection")

        if template is None and self._config.auto_detect_template:
            matches = self._template_library.match(raw_texts, region.bounds)
            if matches and matches[0].confidence > self._config.min_confidence:
                template = matches[0].template
                logger.debug(f"Auto-selected template: {template.name} (confidence: {matches[0].confidence:.2f})")

        # 4. Extract fields
        extraction = self._field_extractor.extract(
            entities,
            region.bounds,
            template,
        )

        # 5. Build metadata
        metadata = self._build_metadata(extraction, region, template)

        return metadata

    def parse_from_bytes(
        self,
        file_bytes: bytes,
        template_name: Optional[str] = None,
    ) -> TitleblockMetadata:
        """
        Parse titleblock from file bytes.

        Args:
            file_bytes: DXF file content as bytes
            template_name: Optional template to use

        Returns:
            TitleblockMetadata
        """
        import tempfile

        try:
            import ezdxf
        except ImportError:
            raise ImportError("ezdxf is required for titleblock parsing")

        # Write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        try:
            return self.parse(tmp_path, template_name)
        finally:
            tmp_path.unlink()

    def _extract_text_labels(
        self,
        entities: List[Any],
        region: BoundingBox,
    ) -> List[str]:
        """Extract text labels for template matching."""
        labels = []

        for entity in entities:
            try:
                entity_type = entity.dxftype() if hasattr(entity, "dxftype") else ""

                if entity_type in ("TEXT", "MTEXT"):
                    pos = entity.dxf.insert if hasattr(entity.dxf, "insert") else None
                    if pos and region.contains_point(pos.x, pos.y):
                        text = entity.dxf.text if entity_type == "TEXT" else entity.text
                        if text:
                            labels.append(text.strip())

                elif entity_type == "ATTRIB":
                    pos = entity.dxf.insert if hasattr(entity.dxf, "insert") else None
                    if pos and region.contains_point(pos.x, pos.y):
                        tag = entity.dxf.tag if hasattr(entity.dxf, "tag") else ""
                        if tag:
                            labels.append(tag.strip())

                elif entity_type == "INSERT":
                    for attrib in entity.attribs:
                        pos = attrib.dxf.insert
                        if region.contains_point(pos.x, pos.y):
                            tag = attrib.dxf.tag if hasattr(attrib.dxf, "tag") else ""
                            if tag:
                                labels.append(tag.strip())

            except Exception:
                continue

        return labels

    def _build_metadata(
        self,
        extraction: ExtractionResult,
        region: TitleblockRegion,
        template: Optional[TitleblockTemplate],
    ) -> TitleblockMetadata:
        """Build metadata from extraction result."""
        metadata = TitleblockMetadata(
            confidence=extraction.confidence,
            template_name=template.name if template else None,
            detection_method=region.method.value,
            region_bounds=region.bounds.to_dict(),
        )

        # Map extracted fields to metadata
        field_mapping = {
            FieldType.PART_NUMBER: "part_number",
            FieldType.DRAWING_TITLE: "drawing_title",
            FieldType.MATERIAL: "material",
            FieldType.AUTHOR: "author",
            FieldType.CHECKER: "checker",
            FieldType.APPROVER: "approver",
            FieldType.DATE: "date",
            FieldType.REVISION: "revision",
            FieldType.SCALE: "scale",
            FieldType.SHEET: "sheet",
            FieldType.WEIGHT: "weight",
            FieldType.COMPANY: "company",
            FieldType.PROJECT: "project",
        }

        for field_item in extraction.fields:
            if field_item.field_type in field_mapping:
                setattr(metadata, field_mapping[field_item.field_type], field_item.value)
            elif field_item.field_type == FieldType.CUSTOM:
                metadata.custom_fields[field_item.label] = field_item.value

        return metadata

    def get_available_templates(self) -> List[str]:
        """Get list of available template names."""
        return self._template_library.template_names

    def add_custom_template(self, template: TitleblockTemplate) -> None:
        """Add a custom template."""
        self._template_library.add_template(template)


# Global parser instance
_default_parser: Optional[TitleblockParser] = None


def get_titleblock_parser() -> TitleblockParser:
    """Get default titleblock parser."""
    global _default_parser
    if _default_parser is None:
        _default_parser = TitleblockParser()
    return _default_parser


def parse_titleblock(
    dxf_path: Path,
    template_name: Optional[str] = None,
) -> TitleblockMetadata:
    """
    Convenience function to parse titleblock.

    Args:
        dxf_path: Path to DXF file
        template_name: Optional template name

    Returns:
        TitleblockMetadata
    """
    parser = get_titleblock_parser()
    return parser.parse(dxf_path, template_name)
