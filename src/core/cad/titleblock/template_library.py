"""
Titleblock template library.

Provides:
- Standard template definitions (ISO, GB, custom)
- Template matching
- Field position mappings
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.cad.titleblock.region_detector import BoundingBox

logger = logging.getLogger(__name__)


class FieldType(str, Enum):
    """Type of titleblock field."""
    PART_NUMBER = "part_number"
    DRAWING_TITLE = "drawing_title"
    MATERIAL = "material"
    AUTHOR = "author"
    CHECKER = "checker"
    APPROVER = "approver"
    DATE = "date"
    REVISION = "revision"
    SCALE = "scale"
    SHEET = "sheet"
    WEIGHT = "weight"
    SURFACE_FINISH = "surface_finish"
    TOLERANCE = "tolerance"
    PROJECT = "project"
    COMPANY = "company"
    CUSTOM = "custom"


@dataclass
class FieldDefinition:
    """Definition of a field in a titleblock template."""
    field_type: FieldType
    name: str  # Display name
    aliases: List[str] = field(default_factory=list)  # Alternative names/labels
    relative_position: Optional[Tuple[float, float]] = None  # (x_ratio, y_ratio) from bottom-left
    relative_size: Optional[Tuple[float, float]] = None  # (width_ratio, height_ratio)
    pattern: Optional[str] = None  # Regex pattern for validation
    required: bool = False
    default_value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_type": self.field_type.value,
            "name": self.name,
            "aliases": self.aliases,
            "relative_position": self.relative_position,
            "relative_size": self.relative_size,
            "pattern": self.pattern,
            "required": self.required,
            "default_value": self.default_value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldDefinition":
        return cls(
            field_type=FieldType(data["field_type"]),
            name=data["name"],
            aliases=data.get("aliases", []),
            relative_position=tuple(data["relative_position"]) if data.get("relative_position") else None,
            relative_size=tuple(data["relative_size"]) if data.get("relative_size") else None,
            pattern=data.get("pattern"),
            required=data.get("required", False),
            default_value=data.get("default_value"),
        )


@dataclass
class TitleblockTemplate:
    """Template for a titleblock format."""
    name: str
    standard: str  # e.g., "ISO", "GB", "ANSI", "Custom"
    description: str = ""
    aspect_ratio: Optional[float] = None  # width/height ratio
    fields: List[FieldDefinition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_field(self, field_type: FieldType) -> Optional[FieldDefinition]:
        """Get field definition by type."""
        for f in self.fields:
            if f.field_type == field_type:
                return f
        return None

    def get_required_fields(self) -> List[FieldDefinition]:
        """Get all required fields."""
        return [f for f in self.fields if f.required]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "standard": self.standard,
            "description": self.description,
            "aspect_ratio": self.aspect_ratio,
            "fields": [f.to_dict() for f in self.fields],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TitleblockTemplate":
        return cls(
            name=data["name"],
            standard=data["standard"],
            description=data.get("description", ""),
            aspect_ratio=data.get("aspect_ratio"),
            fields=[FieldDefinition.from_dict(f) for f in data.get("fields", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass
class TemplateMatch:
    """Result of template matching."""
    template: TitleblockTemplate
    confidence: float
    matched_fields: int
    total_fields: int
    aspect_ratio_diff: Optional[float] = None

    @property
    def match_ratio(self) -> float:
        return self.matched_fields / self.total_fields if self.total_fields > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_name": self.template.name,
            "standard": self.template.standard,
            "confidence": round(self.confidence, 4),
            "matched_fields": self.matched_fields,
            "total_fields": self.total_fields,
            "match_ratio": round(self.match_ratio, 4),
            "aspect_ratio_diff": round(self.aspect_ratio_diff, 4) if self.aspect_ratio_diff else None,
        }


class TemplateLibrary:
    """
    Library of titleblock templates.

    Provides:
    - Standard template definitions
    - Template matching
    - Custom template registration
    """

    def __init__(self):
        """Initialize template library with standard templates."""
        self._templates: Dict[str, TitleblockTemplate] = {}
        self._load_standard_templates()

    def _load_standard_templates(self) -> None:
        """Load standard templates."""
        # ISO 7200 Standard Template
        iso_template = TitleblockTemplate(
            name="ISO_7200",
            standard="ISO",
            description="ISO 7200 standard titleblock for technical drawings",
            aspect_ratio=3.5,
            fields=[
                FieldDefinition(
                    field_type=FieldType.DRAWING_TITLE,
                    name="Title",
                    aliases=["TITLE", "标题", "图名", "NAME"],
                    required=True,
                ),
                FieldDefinition(
                    field_type=FieldType.PART_NUMBER,
                    name="Drawing Number",
                    aliases=["DWG NO", "PART NO", "图号", "零件号", "编号", "NUMBER"],
                    required=True,
                ),
                FieldDefinition(
                    field_type=FieldType.REVISION,
                    name="Revision",
                    aliases=["REV", "版本", "修订"],
                    required=False,
                ),
                FieldDefinition(
                    field_type=FieldType.SCALE,
                    name="Scale",
                    aliases=["SCALE", "比例"],
                    pattern=r"\d+:\d+",
                    required=False,
                ),
                FieldDefinition(
                    field_type=FieldType.DATE,
                    name="Date",
                    aliases=["DATE", "日期"],
                    required=False,
                ),
                FieldDefinition(
                    field_type=FieldType.AUTHOR,
                    name="Drawn By",
                    aliases=["DRAWN", "制图", "绘图"],
                    required=False,
                ),
                FieldDefinition(
                    field_type=FieldType.CHECKER,
                    name="Checked By",
                    aliases=["CHECKED", "校对", "审核"],
                    required=False,
                ),
                FieldDefinition(
                    field_type=FieldType.APPROVER,
                    name="Approved By",
                    aliases=["APPROVED", "批准", "审批"],
                    required=False,
                ),
                FieldDefinition(
                    field_type=FieldType.MATERIAL,
                    name="Material",
                    aliases=["MATERIAL", "材料", "MAT"],
                    required=False,
                ),
                FieldDefinition(
                    field_type=FieldType.SHEET,
                    name="Sheet",
                    aliases=["SHEET", "张", "页"],
                    pattern=r"\d+/\d+",
                    required=False,
                ),
            ],
        )
        self._templates["ISO_7200"] = iso_template

        # GB/T 10609 Chinese Standard Template
        gb_template = TitleblockTemplate(
            name="GB_T_10609",
            standard="GB",
            description="GB/T 10609 中国国家标准标题栏",
            aspect_ratio=4.0,
            fields=[
                FieldDefinition(
                    field_type=FieldType.DRAWING_TITLE,
                    name="图名",
                    aliases=["TITLE", "名称", "图纸名称"],
                    required=True,
                ),
                FieldDefinition(
                    field_type=FieldType.PART_NUMBER,
                    name="图号",
                    aliases=["图样代号", "代号", "编号", "DWG NO"],
                    required=True,
                ),
                FieldDefinition(
                    field_type=FieldType.MATERIAL,
                    name="材料",
                    aliases=["MATERIAL", "材质", "MAT"],
                    required=False,
                ),
                FieldDefinition(
                    field_type=FieldType.SCALE,
                    name="比例",
                    aliases=["SCALE"],
                    pattern=r"\d+:\d+",
                    required=False,
                ),
                FieldDefinition(
                    field_type=FieldType.WEIGHT,
                    name="重量",
                    aliases=["WEIGHT", "质量"],
                    required=False,
                ),
                FieldDefinition(
                    field_type=FieldType.AUTHOR,
                    name="制图",
                    aliases=["DRAWN", "绘制"],
                    required=False,
                ),
                FieldDefinition(
                    field_type=FieldType.CHECKER,
                    name="校对",
                    aliases=["CHECKED", "审核"],
                    required=False,
                ),
                FieldDefinition(
                    field_type=FieldType.APPROVER,
                    name="批准",
                    aliases=["APPROVED", "审批"],
                    required=False,
                ),
                FieldDefinition(
                    field_type=FieldType.DATE,
                    name="日期",
                    aliases=["DATE"],
                    required=False,
                ),
                FieldDefinition(
                    field_type=FieldType.SHEET,
                    name="张数",
                    aliases=["SHEET", "共张"],
                    required=False,
                ),
                FieldDefinition(
                    field_type=FieldType.COMPANY,
                    name="单位",
                    aliases=["COMPANY", "公司", "企业"],
                    required=False,
                ),
            ],
        )
        self._templates["GB_T_10609"] = gb_template

        # Simple template for basic titleblocks
        simple_template = TitleblockTemplate(
            name="Simple",
            standard="Custom",
            description="Simple titleblock with minimal fields",
            aspect_ratio=3.0,
            fields=[
                FieldDefinition(
                    field_type=FieldType.DRAWING_TITLE,
                    name="Title",
                    aliases=["TITLE", "NAME", "标题", "图名", "名称"],
                    required=True,
                ),
                FieldDefinition(
                    field_type=FieldType.PART_NUMBER,
                    name="Number",
                    aliases=["NO", "NUM", "图号", "编号", "DWG"],
                    required=False,
                ),
                FieldDefinition(
                    field_type=FieldType.DATE,
                    name="Date",
                    aliases=["DATE", "日期"],
                    required=False,
                ),
            ],
        )
        self._templates["Simple"] = simple_template

    @property
    def template_names(self) -> List[str]:
        """Get all template names."""
        return list(self._templates.keys())

    def get_template(self, name: str) -> Optional[TitleblockTemplate]:
        """Get template by name."""
        return self._templates.get(name)

    def add_template(self, template: TitleblockTemplate) -> None:
        """Add a template to the library."""
        self._templates[template.name] = template
        logger.info(f"Added template: {template.name}")

    def remove_template(self, name: str) -> bool:
        """Remove a template from the library."""
        if name in self._templates:
            del self._templates[name]
            return True
        return False

    def match(
        self,
        detected_labels: List[str],
        region_bounds: Optional[BoundingBox] = None,
        top_k: int = 3,
    ) -> List[TemplateMatch]:
        """
        Match detected labels against templates.

        Args:
            detected_labels: List of text labels found in titleblock
            region_bounds: Detected titleblock bounds (for aspect ratio matching)
            top_k: Number of top matches to return

        Returns:
            List of TemplateMatch sorted by confidence
        """
        matches = []
        detected_labels_lower = [label.lower().strip() for label in detected_labels]

        for template in self._templates.values():
            matched_fields = 0
            total_fields = len(template.fields)

            for field_def in template.fields:
                # Check if any alias matches
                field_names = [field_def.name.lower()] + [a.lower() for a in field_def.aliases]

                for label in detected_labels_lower:
                    if any(fn in label or label in fn for fn in field_names):
                        matched_fields += 1
                        break

            # Calculate aspect ratio difference if available
            aspect_ratio_diff = None
            if region_bounds and template.aspect_ratio:
                actual_ratio = region_bounds.width / region_bounds.height if region_bounds.height > 0 else 0
                aspect_ratio_diff = abs(actual_ratio - template.aspect_ratio)

            # Calculate confidence
            match_score = matched_fields / total_fields if total_fields > 0 else 0
            aspect_score = 1.0 - min(1.0, aspect_ratio_diff / 2.0) if aspect_ratio_diff else 0.5
            confidence = match_score * 0.7 + aspect_score * 0.3

            matches.append(TemplateMatch(
                template=template,
                confidence=confidence,
                matched_fields=matched_fields,
                total_fields=total_fields,
                aspect_ratio_diff=aspect_ratio_diff,
            ))

        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches[:top_k]

    def save_to_file(self, path: Path) -> None:
        """Save templates to JSON file."""
        data = {name: template.to_dict() for name, template in self._templates.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self._templates)} templates to {path}")

    def load_from_file(self, path: Path) -> int:
        """
        Load templates from JSON file.

        Returns:
            Number of templates loaded
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for name, template_data in data.items():
            try:
                template = TitleblockTemplate.from_dict(template_data)
                self._templates[name] = template
                count += 1
            except Exception as e:
                logger.error(f"Failed to load template {name}: {e}")

        logger.info(f"Loaded {count} templates from {path}")
        return count
