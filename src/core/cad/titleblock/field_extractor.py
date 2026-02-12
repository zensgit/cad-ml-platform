"""
Field extraction from titleblock regions.

Provides:
- Text entity extraction
- Label-value pair detection
- OCR integration
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.core.cad.titleblock.region_detector import BoundingBox
from src.core.cad.titleblock.template_library import FieldDefinition, FieldType, TitleblockTemplate

logger = logging.getLogger(__name__)


@dataclass
class ExtractedField:
    """An extracted field from the titleblock."""
    field_type: FieldType
    label: str
    value: str
    confidence: float
    position: Optional[Tuple[float, float]] = None
    source: str = "text"  # "text", "attrib", "ocr"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_type": self.field_type.value,
            "label": self.label,
            "value": self.value,
            "confidence": round(self.confidence, 4),
            "position": self.position,
            "source": self.source,
        }


@dataclass
class ExtractionResult:
    """Result of field extraction."""
    fields: List[ExtractedField]
    raw_texts: List[Dict[str, Any]]
    unmatched_texts: List[str]
    confidence: float

    def get_field(self, field_type: FieldType) -> Optional[ExtractedField]:
        """Get field by type."""
        for f in self.fields:
            if f.field_type == field_type:
                return f
        return None

    def get_value(self, field_type: FieldType) -> Optional[str]:
        """Get field value by type."""
        field = self.get_field(field_type)
        return field.value if field else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fields": [f.to_dict() for f in self.fields],
            "raw_texts_count": len(self.raw_texts),
            "unmatched_count": len(self.unmatched_texts),
            "confidence": round(self.confidence, 4),
        }


class FieldExtractor:
    """
    Extractor for titleblock fields.

    Supports:
    - DXF text entity extraction
    - ATTRIB value extraction
    - Label-value pair detection
    - OCR result integration
    """

    def __init__(
        self,
        label_value_patterns: Optional[List[str]] = None,
        min_confidence: float = 0.5,
    ):
        """
        Initialize field extractor.

        Args:
            label_value_patterns: Regex patterns for label:value extraction
            min_confidence: Minimum confidence for field matching
        """
        self._label_value_patterns = label_value_patterns or [
            r"^(.+?)[:\uff1a]\s*(.+)$",  # label: value or label：value (Chinese colon)
            r"^(.+?)\s{2,}(.+)$",  # label  value (multiple spaces)
        ]
        self._min_confidence = min_confidence

    def extract(
        self,
        entities: List[Any],
        region: BoundingBox,
        template: Optional[TitleblockTemplate] = None,
    ) -> ExtractionResult:
        """
        Extract fields from entities in a region.

        Args:
            entities: List of DXF entities
            region: Titleblock region bounds
            template: Template for field matching

        Returns:
            ExtractionResult
        """
        # Extract raw text data
        raw_texts = self._extract_raw_texts(entities, region)

        # Parse label-value pairs
        parsed_pairs = self._parse_label_value_pairs(raw_texts)

        # Match to template fields
        if template:
            fields = self._match_to_template(parsed_pairs, raw_texts, template)
        else:
            fields = self._auto_detect_fields(parsed_pairs, raw_texts)

        # Calculate overall confidence
        if fields:
            confidence = sum(f.confidence for f in fields) / len(fields)
        else:
            confidence = 0.0

        # Find unmatched texts
        matched_values = {f.value for f in fields}
        unmatched = [
            t["text"] for t in raw_texts
            if t["text"] not in matched_values
        ]

        return ExtractionResult(
            fields=fields,
            raw_texts=raw_texts,
            unmatched_texts=unmatched,
            confidence=confidence,
        )

    def _extract_raw_texts(
        self,
        entities: List[Any],
        region: BoundingBox,
    ) -> List[Dict[str, Any]]:
        """Extract raw text data from entities in region."""
        texts = []

        for entity in entities:
            try:
                entity_type = entity.dxftype() if hasattr(entity, "dxftype") else ""

                if entity_type == "TEXT":
                    pos = entity.dxf.insert
                    if region.contains_point(pos.x, pos.y):
                        texts.append({
                            "text": entity.dxf.text.strip(),
                            "position": (pos.x, pos.y),
                            "height": entity.dxf.height,
                            "source": "text",
                        })

                elif entity_type == "MTEXT":
                    pos = entity.dxf.insert
                    if region.contains_point(pos.x, pos.y):
                        # MTEXT can have formatting, extract plain text
                        text = entity.text
                        # Remove formatting codes
                        text = re.sub(r"\\[A-Za-z][^;]*;", "", text)
                        text = re.sub(r"\{|\}", "", text)
                        texts.append({
                            "text": text.strip(),
                            "position": (pos.x, pos.y),
                            "height": entity.dxf.char_height if hasattr(entity.dxf, "char_height") else 2.5,
                            "source": "mtext",
                        })

                elif entity_type == "ATTRIB":
                    pos = entity.dxf.insert
                    if region.contains_point(pos.x, pos.y):
                        texts.append({
                            "text": entity.dxf.text.strip(),
                            "tag": entity.dxf.tag if hasattr(entity.dxf, "tag") else "",
                            "position": (pos.x, pos.y),
                            "height": entity.dxf.height if hasattr(entity.dxf, "height") else 2.5,
                            "source": "attrib",
                        })

                elif entity_type == "INSERT":
                    # Block references may contain ATTRIBs
                    for attrib in entity.attribs:
                        pos = attrib.dxf.insert
                        if region.contains_point(pos.x, pos.y):
                            texts.append({
                                "text": attrib.dxf.text.strip(),
                                "tag": attrib.dxf.tag if hasattr(attrib.dxf, "tag") else "",
                                "position": (pos.x, pos.y),
                                "height": attrib.dxf.height if hasattr(attrib.dxf, "height") else 2.5,
                                "source": "attrib",
                            })

            except Exception as e:
                logger.debug(f"Error extracting text from entity: {e}")
                continue

        return texts

    def _parse_label_value_pairs(
        self,
        raw_texts: List[Dict[str, Any]],
    ) -> List[Tuple[str, str, float, float]]:
        """
        Parse label-value pairs from raw texts.

        Returns list of (label, value, x, y) tuples.
        """
        pairs = []

        for text_data in raw_texts:
            text = text_data["text"]
            pos = text_data.get("position", (0, 0))

            # Skip empty or very short texts
            if len(text) < 2:
                continue

            # Try each pattern
            for pattern in self._label_value_patterns:
                match = re.match(pattern, text)
                if match:
                    label = match.group(1).strip()
                    value = match.group(2).strip()
                    if label and value:
                        pairs.append((label, value, pos[0], pos[1]))
                        break
            else:
                # No pattern matched, treat as potential standalone label or value
                # Check if it looks like a label (short, ends with common label suffixes)
                if self._looks_like_label(text):
                    # Look for nearby value
                    pairs.append((text, "", pos[0], pos[1]))
                else:
                    # Treat as potential value
                    pairs.append(("", text, pos[0], pos[1]))

        return pairs

    def _looks_like_label(self, text: str) -> bool:
        """Check if text looks like a field label."""
        # Short text
        if len(text) > 20:
            return False

        # Common label patterns
        label_indicators = [
            ":", "：", "号", "名", "日期", "比例", "材料",
            "制图", "审核", "批准", "DATE", "SCALE", "DRAWN",
            "TITLE", "MATERIAL", "REV", "NO"
        ]

        return any(ind in text.upper() for ind in label_indicators)

    def _match_to_template(
        self,
        pairs: List[Tuple[str, str, float, float]],
        raw_texts: List[Dict[str, Any]],
        template: TitleblockTemplate,
    ) -> List[ExtractedField]:
        """Match parsed pairs to template fields."""
        fields = []
        used_values = set()

        for field_def in template.fields:
            # Build list of possible labels for this field
            possible_labels = [field_def.name.lower()] + [a.lower() for a in field_def.aliases]

            # Find matching pair
            best_match = None
            best_confidence = 0.0

            for label, value, x, y in pairs:
                label_lower = label.lower()

                # Check label match
                label_score = 0.0
                for possible in possible_labels:
                    if possible in label_lower or label_lower in possible:
                        label_score = 1.0 if possible == label_lower else 0.8
                        break

                if label_score > 0 and value and value not in used_values:
                    # Validate with pattern if available
                    pattern_score = 1.0
                    if field_def.pattern:
                        if not re.match(field_def.pattern, value):
                            pattern_score = 0.5

                    confidence = label_score * pattern_score
                    if confidence > best_confidence:
                        best_match = (label, value, x, y)
                        best_confidence = confidence

            # If no label match, try to find value by position or pattern
            if best_match is None and field_def.pattern:
                for text_data in raw_texts:
                    text = text_data["text"]
                    if text not in used_values and re.match(field_def.pattern, text):
                        pos = text_data.get("position", (0, 0))
                        best_match = (field_def.name, text, pos[0], pos[1])
                        best_confidence = 0.6
                        break

            if best_match:
                fields.append(ExtractedField(
                    field_type=field_def.field_type,
                    label=best_match[0],
                    value=best_match[1],
                    confidence=best_confidence,
                    position=(best_match[2], best_match[3]),
                    source="text",
                ))
                used_values.add(best_match[1])

        return fields

    def _auto_detect_fields(
        self,
        pairs: List[Tuple[str, str, float, float]],
        raw_texts: List[Dict[str, Any]],
    ) -> List[ExtractedField]:
        """Auto-detect fields without template."""
        fields = []

        # Common field patterns
        field_patterns = {
            FieldType.PART_NUMBER: [
                r"^[A-Z]{2,}\d{4,}.*$",  # e.g., DWG12345
                r"^\d{4,}-\d{2,}.*$",  # e.g., 12345-01
            ],
            FieldType.SCALE: [
                r"^\d+:\d+$",  # e.g., 1:1, 1:10
            ],
            FieldType.DATE: [
                r"^\d{4}[-/]\d{2}[-/]\d{2}$",  # e.g., 2024-01-01
                r"^\d{2}[-/]\d{2}[-/]\d{4}$",  # e.g., 01/01/2024
            ],
            FieldType.SHEET: [
                r"^\d+/\d+$",  # e.g., 1/1, 2/5
            ],
            FieldType.REVISION: [
                r"^[A-Z]$",  # e.g., A, B, C
                r"^REV[.-]?\s*[A-Z0-9]+$",  # e.g., REV-A, REV.01
            ],
        }

        # Label keywords
        label_keywords = {
            FieldType.DRAWING_TITLE: ["title", "名称", "标题", "图名", "name"],
            FieldType.PART_NUMBER: ["no", "number", "图号", "编号", "代号", "dwg"],
            FieldType.MATERIAL: ["material", "材料", "材质", "mat"],
            FieldType.AUTHOR: ["drawn", "制图", "绘制", "绘图"],
            FieldType.CHECKER: ["checked", "校对", "审核"],
            FieldType.APPROVER: ["approved", "批准", "审批"],
            FieldType.DATE: ["date", "日期"],
            FieldType.SCALE: ["scale", "比例"],
            FieldType.WEIGHT: ["weight", "重量", "质量"],
        }

        used_values = set()

        # First pass: match by label keywords
        for label, value, x, y in pairs:
            if not value or value in used_values:
                continue

            label_lower = label.lower()
            for field_type, keywords in label_keywords.items():
                if any(kw in label_lower for kw in keywords):
                    fields.append(ExtractedField(
                        field_type=field_type,
                        label=label,
                        value=value,
                        confidence=0.8,
                        position=(x, y),
                    ))
                    used_values.add(value)
                    break

        # Second pass: match by value patterns
        for text_data in raw_texts:
            text = text_data["text"]
            if text in used_values or len(text) < 2:
                continue

            pos = text_data.get("position", (0, 0))

            for field_type, patterns in field_patterns.items():
                # Skip if already found this field type
                if any(f.field_type == field_type for f in fields):
                    continue

                for pattern in patterns:
                    if re.match(pattern, text, re.IGNORECASE):
                        fields.append(ExtractedField(
                            field_type=field_type,
                            label="",
                            value=text,
                            confidence=0.6,
                            position=pos,
                        ))
                        used_values.add(text)
                        break

        return fields

    def integrate_ocr_results(
        self,
        extraction_result: ExtractionResult,
        ocr_texts: List[Dict[str, Any]],
        region: BoundingBox,
    ) -> ExtractionResult:
        """
        Integrate OCR results with extracted fields.

        Args:
            extraction_result: Existing extraction result
            ocr_texts: OCR results with text, bbox, confidence
            region: Titleblock region bounds

        Returns:
            Updated ExtractionResult
        """
        # Filter OCR texts in region
        relevant_ocr = []
        for ocr_item in ocr_texts:
            bbox = ocr_item.get("bbox", [])
            if len(bbox) >= 4:
                # Check if center is in region
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                if region.contains_point(center_x, center_y):
                    relevant_ocr.append(ocr_item)

        # Add OCR texts to raw texts
        for ocr_item in relevant_ocr:
            bbox = ocr_item.get("bbox", [0, 0, 0, 0])
            extraction_result.raw_texts.append({
                "text": ocr_item.get("text", ""),
                "position": ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                "confidence": ocr_item.get("confidence", 0.5),
                "source": "ocr",
            })

        # Try to fill missing fields with OCR
        existing_types = {f.field_type for f in extraction_result.fields}
        pairs = self._parse_label_value_pairs(extraction_result.raw_texts)
        new_fields = self._auto_detect_fields(pairs, extraction_result.raw_texts)

        for field_item in new_fields:
            if field_item.field_type not in existing_types:
                field_item.source = "ocr"
                field_item.confidence *= 0.8  # Reduce confidence for OCR
                extraction_result.fields.append(field_item)

        # Recalculate confidence
        if extraction_result.fields:
            extraction_result.confidence = sum(f.confidence for f in extraction_result.fields) / len(extraction_result.fields)

        return extraction_result
