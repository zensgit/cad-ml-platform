"""
ATTRIB attribute extraction for DXF files.

Extracts attributes from block references and titleblocks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class AttributeDefinition:
    """ATTDEF definition from block."""
    tag: str
    prompt: str
    default_value: str
    insert_point: Tuple[float, float, float]
    height: float
    rotation: float
    layer: str
    is_invisible: bool = False
    is_constant: bool = False
    is_verify: bool = False
    is_preset: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag": self.tag,
            "prompt": self.prompt,
            "default_value": self.default_value,
            "insert_point": self.insert_point,
            "height": self.height,
            "rotation": self.rotation,
            "layer": self.layer,
            "is_invisible": self.is_invisible,
            "is_constant": self.is_constant,
        }


@dataclass
class ATTRIBData:
    """ATTRIB data from block reference."""
    tag: str
    value: str
    insert_point: Tuple[float, float, float]
    height: float
    rotation: float
    layer: str
    style: str
    block_name: str
    block_handle: str
    is_invisible: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag": self.tag,
            "value": self.value,
            "insert_point": self.insert_point,
            "height": self.height,
            "rotation": self.rotation,
            "layer": self.layer,
            "block_name": self.block_name,
            "is_invisible": self.is_invisible,
        }


@dataclass
class TitleblockData:
    """Extracted titleblock information."""
    block_name: str
    drawing_number: Optional[str] = None
    drawing_title: Optional[str] = None
    revision: Optional[str] = None
    date: Optional[str] = None
    drawn_by: Optional[str] = None
    checked_by: Optional[str] = None
    approved_by: Optional[str] = None
    scale: Optional[str] = None
    sheet: Optional[str] = None
    material: Optional[str] = None
    finish: Optional[str] = None
    weight: Optional[str] = None
    custom_fields: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "block_name": self.block_name,
            "drawing_number": self.drawing_number,
            "drawing_title": self.drawing_title,
            "revision": self.revision,
            "date": self.date,
            "drawn_by": self.drawn_by,
            "checked_by": self.checked_by,
            "approved_by": self.approved_by,
            "scale": self.scale,
            "sheet": self.sheet,
            "material": self.material,
            "finish": self.finish,
            "weight": self.weight,
        }
        result.update(self.custom_fields)
        return {k: v for k, v in result.items() if v is not None}


class AttributeExtractor:
    """
    Extracts attributes from DXF block references.

    Handles ATTDEF definitions and ATTRIB values.
    """

    # Common titleblock field patterns
    TITLEBLOCK_PATTERNS = {
        "drawing_number": [
            r"(?i)^(dwg|drawing)[-_\s]?(no|num|number)?$",
            r"(?i)^part[-_\s]?(no|num|number)?$",
            r"(?i)^doc[-_\s]?(no|num|number)?$",
        ],
        "drawing_title": [
            r"(?i)^title$",
            r"(?i)^(dwg|drawing)[-_\s]?title$",
            r"(?i)^description$",
            r"(?i)^name$",
        ],
        "revision": [
            r"(?i)^rev(ision)?$",
            r"(?i)^ver(sion)?$",
            r"(?i)^issue$",
        ],
        "date": [
            r"(?i)^date$",
            r"(?i)^(dwg|drawing)[-_\s]?date$",
            r"(?i)^created[-_\s]?date$",
        ],
        "drawn_by": [
            r"(?i)^drawn[-_\s]?by$",
            r"(?i)^drafter$",
            r"(?i)^author$",
            r"(?i)^designer$",
        ],
        "checked_by": [
            r"(?i)^checked[-_\s]?by$",
            r"(?i)^checker$",
            r"(?i)^reviewed[-_\s]?by$",
        ],
        "approved_by": [
            r"(?i)^approved[-_\s]?by$",
            r"(?i)^approver$",
        ],
        "scale": [
            r"(?i)^scale$",
            r"(?i)^(dwg|drawing)[-_\s]?scale$",
        ],
        "sheet": [
            r"(?i)^sheet$",
            r"(?i)^page$",
            r"(?i)^sheet[-_\s]?(no|num|number)?$",
        ],
        "material": [
            r"(?i)^material$",
            r"(?i)^mat(l)?$",
        ],
        "finish": [
            r"(?i)^finish$",
            r"(?i)^surface[-_\s]?finish$",
        ],
        "weight": [
            r"(?i)^weight$",
            r"(?i)^mass$",
        ],
    }

    # Common titleblock block name patterns
    TITLEBLOCK_BLOCK_PATTERNS = [
        r"(?i)title[-_\s]?block",
        r"(?i)border",
        r"(?i)frame",
        r"(?i)^A[0-4]$",  # Paper sizes
        r"(?i)^ANSI",
        r"(?i)^ISO",
        r"(?i)^DIN",
    ]

    def __init__(self):
        """Initialize attribute extractor."""
        self._definitions: Dict[str, List[AttributeDefinition]] = {}
        self._attributes: List[ATTRIBData] = []

    @property
    def definitions(self) -> Dict[str, List[AttributeDefinition]]:
        return self._definitions.copy()

    @property
    def attributes(self) -> List[ATTRIBData]:
        return self._attributes.copy()

    def extract_definitions(self, dxf_doc: Any) -> Dict[str, List[AttributeDefinition]]:
        """
        Extract attribute definitions from blocks.

        Args:
            dxf_doc: ezdxf document

        Returns:
            Dict mapping block name to list of AttributeDefinition
        """
        self._definitions.clear()

        if hasattr(dxf_doc, "blocks"):
            for block in dxf_doc.blocks:
                attdefs = []
                for entity in block:
                    if entity.dxftype() == "ATTDEF":
                        attdef = self._create_attdef(entity)
                        attdefs.append(attdef)

                if attdefs:
                    self._definitions[block.name] = attdefs

        logger.info(f"Found attribute definitions in {len(self._definitions)} blocks")
        return self._definitions

    def _create_attdef(self, entity: Any) -> AttributeDefinition:
        """Create AttributeDefinition from ATTDEF entity."""
        insert = entity.dxf.insert
        return AttributeDefinition(
            tag=entity.dxf.tag,
            prompt=getattr(entity.dxf, "prompt", ""),
            default_value=getattr(entity.dxf, "text", ""),
            insert_point=(insert.x, insert.y, getattr(insert, "z", 0.0)),
            height=entity.dxf.height,
            rotation=getattr(entity.dxf, "rotation", 0.0),
            layer=entity.dxf.layer,
            is_invisible=bool(entity.dxf.flags & 1),
            is_constant=bool(entity.dxf.flags & 2),
            is_verify=bool(entity.dxf.flags & 4),
            is_preset=bool(entity.dxf.flags & 8),
        )

    def extract_attributes(self, dxf_doc: Any) -> List[ATTRIBData]:
        """
        Extract attributes from block references.

        Args:
            dxf_doc: ezdxf document

        Returns:
            List of ATTRIBData
        """
        self._attributes.clear()

        if hasattr(dxf_doc, "modelspace"):
            for entity in dxf_doc.modelspace():
                if entity.dxftype() == "INSERT":
                    if hasattr(entity, "attribs"):
                        for attrib in entity.attribs:
                            attr_data = self._create_attrib(attrib, entity)
                            self._attributes.append(attr_data)

        # Also check paperspace
        if hasattr(dxf_doc, "layouts"):
            for layout in dxf_doc.layouts:
                if layout.name != "Model":
                    for entity in layout:
                        if entity.dxftype() == "INSERT":
                            if hasattr(entity, "attribs"):
                                for attrib in entity.attribs:
                                    attr_data = self._create_attrib(attrib, entity)
                                    self._attributes.append(attr_data)

        logger.info(f"Extracted {len(self._attributes)} attributes")
        return self._attributes

    def _create_attrib(self, attrib: Any, insert: Any) -> ATTRIBData:
        """Create ATTRIBData from ATTRIB entity."""
        insert_point = attrib.dxf.insert
        return ATTRIBData(
            tag=attrib.dxf.tag,
            value=attrib.dxf.text,
            insert_point=(insert_point.x, insert_point.y, getattr(insert_point, "z", 0.0)),
            height=attrib.dxf.height,
            rotation=getattr(attrib.dxf, "rotation", 0.0),
            layer=attrib.dxf.layer,
            style=getattr(attrib.dxf, "style", "Standard"),
            block_name=insert.dxf.name,
            block_handle=insert.dxf.handle,
            is_invisible=bool(attrib.dxf.flags & 1),
        )

    def find_titleblock_block(self, dxf_doc: Any) -> Optional[str]:
        """
        Find the titleblock block name.

        Args:
            dxf_doc: ezdxf document

        Returns:
            Block name or None
        """
        # Check block names
        if hasattr(dxf_doc, "blocks"):
            for block in dxf_doc.blocks:
                for pattern in self.TITLEBLOCK_BLOCK_PATTERNS:
                    if re.search(pattern, block.name):
                        return block.name

        # Check INSERT entities in paperspace
        if hasattr(dxf_doc, "layouts"):
            for layout in dxf_doc.layouts:
                if layout.name != "Model":
                    for entity in layout:
                        if entity.dxftype() == "INSERT":
                            for pattern in self.TITLEBLOCK_BLOCK_PATTERNS:
                                if re.search(pattern, entity.dxf.name):
                                    return entity.dxf.name

        return None

    def extract_titleblock(
        self,
        dxf_doc: Any,
        block_name: Optional[str] = None,
    ) -> Optional[TitleblockData]:
        """
        Extract titleblock information.

        Args:
            dxf_doc: ezdxf document
            block_name: Specific block name (auto-detected if None)

        Returns:
            TitleblockData or None
        """
        # Find titleblock block
        if block_name is None:
            block_name = self.find_titleblock_block(dxf_doc)
            if block_name is None:
                logger.warning("No titleblock block found")
                return None

        # Extract attributes
        self.extract_attributes(dxf_doc)

        # Filter attributes for this block
        block_attrs = [a for a in self._attributes if a.block_name == block_name]
        if not block_attrs:
            logger.warning(f"No attributes found in block: {block_name}")
            return None

        # Map attributes to titleblock fields
        titleblock = TitleblockData(block_name=block_name)

        for attr in block_attrs:
            tag = attr.tag
            value = attr.value.strip()

            if not value:
                continue

            # Try to match known fields
            matched = False
            for field_name, patterns in self.TITLEBLOCK_PATTERNS.items():
                for pattern in patterns:
                    if re.match(pattern, tag):
                        setattr(titleblock, field_name, value)
                        matched = True
                        break
                if matched:
                    break

            # Store as custom field if not matched
            if not matched:
                titleblock.custom_fields[tag] = value

        return titleblock

    def get_attributes_by_tag(self, tag: str) -> List[ATTRIBData]:
        """Get all attributes with given tag."""
        pattern = re.compile(tag, re.IGNORECASE)
        return [a for a in self._attributes if pattern.search(a.tag)]

    def get_attributes_by_block(self, block_name: str) -> List[ATTRIBData]:
        """Get all attributes from given block."""
        return [a for a in self._attributes if a.block_name == block_name]

    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        # Count unique tags
        tags = [a.tag for a in self._attributes]
        unique_tags = set(tags)

        # Count by block
        block_counts: Dict[str, int] = {}
        for attr in self._attributes:
            block_counts[attr.block_name] = block_counts.get(attr.block_name, 0) + 1

        return {
            "total_attributes": len(self._attributes),
            "unique_tags": len(unique_tags),
            "blocks_with_definitions": len(self._definitions),
            "attributes_by_block": block_counts,
            "invisible_attributes": sum(1 for a in self._attributes if a.is_invisible),
        }


def extract_attributes(dxf_source: Union[str, Path, Any]) -> List[ATTRIBData]:
    """
    Convenience function to extract all attributes.

    Args:
        dxf_source: DXF file path or ezdxf document

    Returns:
        List of ATTRIBData
    """
    try:
        import ezdxf
    except ImportError:
        raise ImportError("ezdxf required: pip install ezdxf")

    if isinstance(dxf_source, (str, Path)):
        doc = ezdxf.readfile(str(dxf_source))
    else:
        doc = dxf_source

    extractor = AttributeExtractor()
    return extractor.extract_attributes(doc)


def extract_titleblock_attributes(
    dxf_source: Union[str, Path, Any],
    block_name: Optional[str] = None,
) -> Optional[TitleblockData]:
    """
    Convenience function to extract titleblock data.

    Args:
        dxf_source: DXF file path or ezdxf document
        block_name: Titleblock block name

    Returns:
        TitleblockData or None
    """
    try:
        import ezdxf
    except ImportError:
        raise ImportError("ezdxf required: pip install ezdxf")

    if isinstance(dxf_source, (str, Path)):
        doc = ezdxf.readfile(str(dxf_source))
    else:
        doc = dxf_source

    extractor = AttributeExtractor()
    return extractor.extract_titleblock(doc, block_name)
