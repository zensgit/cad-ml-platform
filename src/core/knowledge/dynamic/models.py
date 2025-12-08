"""
Data models for dynamic knowledge base.

Defines the structure of knowledge entries that can be
dynamically loaded, updated, and persisted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid


class KnowledgeCategory(str, Enum):
    """Categories of mechanical knowledge."""

    MATERIAL = "material"
    PRECISION = "precision"
    STANDARD = "standard"
    FUNCTIONAL = "functional"
    ASSEMBLY = "assembly"
    MANUFACTURING = "manufacturing"
    GEOMETRY = "geometry"
    PART_TYPE = "part_type"


@dataclass
class KnowledgeEntry:
    """Base class for all knowledge entries."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: KnowledgeCategory = KnowledgeCategory.PART_TYPE
    name: str = ""
    chinese_name: str = ""
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    ocr_patterns: List[str] = field(default_factory=list)
    part_hints: Dict[str, float] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 100  # Higher = more important
    source: str = "user"  # "builtin", "user", "imported"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "category": self.category.value if isinstance(self.category, Enum) else self.category,
            "name": self.name,
            "chinese_name": self.chinese_name,
            "description": self.description,
            "keywords": self.keywords,
            "ocr_patterns": self.ocr_patterns,
            "part_hints": self.part_hints,
            "enabled": self.enabled,
            "priority": self.priority,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntry":
        """Create from dictionary."""
        category = data.get("category", "part_type")
        if isinstance(category, str):
            category = KnowledgeCategory(category)

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            category=category,
            name=data.get("name", ""),
            chinese_name=data.get("chinese_name", ""),
            description=data.get("description", ""),
            keywords=data.get("keywords", []),
            ocr_patterns=data.get("ocr_patterns", []),
            part_hints=data.get("part_hints", {}),
            enabled=data.get("enabled", True),
            priority=data.get("priority", 100),
            source=data.get("source", "user"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MaterialRule(KnowledgeEntry):
    """Material-related knowledge rule."""

    category: KnowledgeCategory = field(default=KnowledgeCategory.MATERIAL)
    material_type: str = ""  # "steel", "aluminum", "bronze", etc.
    material_grades: List[str] = field(default_factory=list)
    hardness_range: Optional[Tuple[int, int]] = None
    typical_applications: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "material_type": self.material_type,
            "material_grades": self.material_grades,
            "hardness_range": self.hardness_range,
            "typical_applications": self.typical_applications,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaterialRule":
        base = KnowledgeEntry.from_dict(data)
        return cls(
            id=base.id,
            category=KnowledgeCategory.MATERIAL,
            name=base.name,
            chinese_name=base.chinese_name,
            description=base.description,
            keywords=base.keywords,
            ocr_patterns=base.ocr_patterns,
            part_hints=base.part_hints,
            enabled=base.enabled,
            priority=base.priority,
            source=base.source,
            created_at=base.created_at,
            updated_at=base.updated_at,
            metadata=base.metadata,
            material_type=data.get("material_type", ""),
            material_grades=data.get("material_grades", []),
            hardness_range=tuple(data["hardness_range"]) if data.get("hardness_range") else None,
            typical_applications=data.get("typical_applications", []),
        )


@dataclass
class PrecisionRule(KnowledgeEntry):
    """Precision/tolerance-related knowledge rule."""

    category: KnowledgeCategory = field(default=KnowledgeCategory.PRECISION)
    tolerance_grade: str = ""  # "IT5", "IT6", "IT7", etc.
    surface_roughness_range: Optional[Tuple[float, float]] = None  # Ra range
    gdt_symbols: List[str] = field(default_factory=list)
    fit_types: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "tolerance_grade": self.tolerance_grade,
            "surface_roughness_range": self.surface_roughness_range,
            "gdt_symbols": self.gdt_symbols,
            "fit_types": self.fit_types,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrecisionRule":
        base = KnowledgeEntry.from_dict(data)
        return cls(
            id=base.id,
            category=KnowledgeCategory.PRECISION,
            name=base.name,
            chinese_name=base.chinese_name,
            description=base.description,
            keywords=base.keywords,
            ocr_patterns=base.ocr_patterns,
            part_hints=base.part_hints,
            enabled=base.enabled,
            priority=base.priority,
            source=base.source,
            created_at=base.created_at,
            updated_at=base.updated_at,
            metadata=base.metadata,
            tolerance_grade=data.get("tolerance_grade", ""),
            surface_roughness_range=(
                tuple(data["surface_roughness_range"])
                if data.get("surface_roughness_range") else None
            ),
            gdt_symbols=data.get("gdt_symbols", []),
            fit_types=data.get("fit_types", []),
        )


@dataclass
class StandardRule(KnowledgeEntry):
    """Industry standard-related knowledge rule."""

    category: KnowledgeCategory = field(default=KnowledgeCategory.STANDARD)
    standard_org: str = ""  # "GB", "ISO", "DIN", "ANSI", etc.
    standard_number: str = ""
    designation_pattern: str = ""  # Regex pattern for standard designation
    year: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "standard_org": self.standard_org,
            "standard_number": self.standard_number,
            "designation_pattern": self.designation_pattern,
            "year": self.year,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardRule":
        base = KnowledgeEntry.from_dict(data)
        return cls(
            id=base.id,
            category=KnowledgeCategory.STANDARD,
            name=base.name,
            chinese_name=base.chinese_name,
            description=base.description,
            keywords=base.keywords,
            ocr_patterns=base.ocr_patterns,
            part_hints=base.part_hints,
            enabled=base.enabled,
            priority=base.priority,
            source=base.source,
            created_at=base.created_at,
            updated_at=base.updated_at,
            metadata=base.metadata,
            standard_org=data.get("standard_org", ""),
            standard_number=data.get("standard_number", ""),
            designation_pattern=data.get("designation_pattern", ""),
            year=data.get("year"),
        )


@dataclass
class FunctionalFeatureRule(KnowledgeEntry):
    """Functional feature-related knowledge rule."""

    category: KnowledgeCategory = field(default=KnowledgeCategory.FUNCTIONAL)
    feature_type: str = ""  # "keyway", "spline", "thread", etc.
    typical_parts: List[str] = field(default_factory=list)
    geometric_indicators: Dict[str, Any] = field(default_factory=dict)
    weight: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "feature_type": self.feature_type,
            "typical_parts": self.typical_parts,
            "geometric_indicators": self.geometric_indicators,
            "weight": self.weight,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FunctionalFeatureRule":
        base = KnowledgeEntry.from_dict(data)
        return cls(
            id=base.id,
            category=KnowledgeCategory.FUNCTIONAL,
            name=base.name,
            chinese_name=base.chinese_name,
            description=base.description,
            keywords=base.keywords,
            ocr_patterns=base.ocr_patterns,
            part_hints=base.part_hints,
            enabled=base.enabled,
            priority=base.priority,
            source=base.source,
            created_at=base.created_at,
            updated_at=base.updated_at,
            metadata=base.metadata,
            feature_type=data.get("feature_type", ""),
            typical_parts=data.get("typical_parts", []),
            geometric_indicators=data.get("geometric_indicators", {}),
            weight=data.get("weight", 0.3),
        )


@dataclass
class AssemblyRule(KnowledgeEntry):
    """Assembly relationship knowledge rule."""

    category: KnowledgeCategory = field(default=KnowledgeCategory.ASSEMBLY)
    part_a: str = ""
    part_b: str = ""
    connection_type: str = ""  # "fit", "fastener", "weld", etc.
    typical_fits: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "part_a": self.part_a,
            "part_b": self.part_b,
            "connection_type": self.connection_type,
            "typical_fits": self.typical_fits,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssemblyRule":
        base = KnowledgeEntry.from_dict(data)
        return cls(
            id=base.id,
            category=KnowledgeCategory.ASSEMBLY,
            name=base.name,
            chinese_name=base.chinese_name,
            description=base.description,
            keywords=base.keywords,
            ocr_patterns=base.ocr_patterns,
            part_hints=base.part_hints,
            enabled=base.enabled,
            priority=base.priority,
            source=base.source,
            created_at=base.created_at,
            updated_at=base.updated_at,
            metadata=base.metadata,
            part_a=data.get("part_a", ""),
            part_b=data.get("part_b", ""),
            connection_type=data.get("connection_type", ""),
            typical_fits=data.get("typical_fits", []),
        )


@dataclass
class ManufacturingRule(KnowledgeEntry):
    """Manufacturing process knowledge rule."""

    category: KnowledgeCategory = field(default=KnowledgeCategory.MANUFACTURING)
    process_type: str = ""  # "machining", "heat_treatment", "surface_treatment"
    process_name: str = ""
    surface_finish_range: Optional[Tuple[float, float]] = None
    tolerance_capability: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "process_type": self.process_type,
            "process_name": self.process_name,
            "surface_finish_range": self.surface_finish_range,
            "tolerance_capability": self.tolerance_capability,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManufacturingRule":
        base = KnowledgeEntry.from_dict(data)
        return cls(
            id=base.id,
            category=KnowledgeCategory.MANUFACTURING,
            name=base.name,
            chinese_name=base.chinese_name,
            description=base.description,
            keywords=base.keywords,
            ocr_patterns=base.ocr_patterns,
            part_hints=base.part_hints,
            enabled=base.enabled,
            priority=base.priority,
            source=base.source,
            created_at=base.created_at,
            updated_at=base.updated_at,
            metadata=base.metadata,
            process_type=data.get("process_type", ""),
            process_name=data.get("process_name", ""),
            surface_finish_range=(
                tuple(data["surface_finish_range"])
                if data.get("surface_finish_range") else None
            ),
            tolerance_capability=data.get("tolerance_capability", ""),
        )


@dataclass
class GeometryPattern(KnowledgeEntry):
    """Geometry-based classification pattern."""

    category: KnowledgeCategory = field(default=KnowledgeCategory.GEOMETRY)
    conditions: Dict[str, Any] = field(default_factory=dict)
    # Example conditions:
    # {
    #     "aspect_variance": {"min": 0.2, "max": 0.5},
    #     "sphericity": {"min": 0.6},
    #     "circle_ratio": {"min": 0.1, "max": 0.3},
    # }

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "conditions": self.conditions,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeometryPattern":
        base = KnowledgeEntry.from_dict(data)
        return cls(
            id=base.id,
            category=KnowledgeCategory.GEOMETRY,
            name=base.name,
            chinese_name=base.chinese_name,
            description=base.description,
            keywords=base.keywords,
            ocr_patterns=base.ocr_patterns,
            part_hints=base.part_hints,
            enabled=base.enabled,
            priority=base.priority,
            source=base.source,
            created_at=base.created_at,
            updated_at=base.updated_at,
            metadata=base.metadata,
            conditions=data.get("conditions", {}),
        )

    # Entity types that can be computed as ratios from entity_counts
    ENTITY_RATIO_TYPES = frozenset({
        "circle", "arc", "line", "ellipse", "spline", "polyline",
        "point", "solid", "facet", "surface", "edge", "vertex",
    })

    def matches(
        self,
        geometric_features: Dict[str, float],
        entity_counts: Dict[str, int],
    ) -> bool:
        """Check if geometry matches this pattern's conditions.

        Feature types are handled as follows:
        1. Entity type ratios (circle_ratio, arc_ratio, line_ratio, etc.)
           → Computed from entity_counts as count/total
        2. Derived features (curved_ratio, disk_indicator, straight_ratio, etc.)
           → Read directly from geometric_features (pre-computed by loader)
        3. Base features (sphericity, aspect_variance, complexity_score, etc.)
           → Read directly from geometric_features
        """
        total_entities = sum(entity_counts.values()) if entity_counts else 0

        for feature, condition in self.conditions.items():
            # Determine feature value based on type
            if feature.endswith("_ratio"):
                # Extract prefix before "_ratio"
                prefix = feature[:-6]  # Remove "_ratio" suffix
                if prefix.lower() in self.ENTITY_RATIO_TYPES:
                    # Entity type ratio: compute from entity_counts
                    entity_type = prefix.upper()
                    count = entity_counts.get(entity_type, 0)
                    value = count / total_entities if total_entities > 0 else 0
                else:
                    # Derived ratio (curved_ratio, straight_ratio, etc.): read from features
                    value = geometric_features.get(feature, 0.0)
            elif feature == "total_entities":
                value = total_entities
            elif feature.endswith("_count"):
                # Entity count features (circle_count, arc_count, etc.)
                prefix = feature[:-6].upper()  # Remove "_count" suffix
                value = float(entity_counts.get(prefix, 0))
            else:
                # Base geometric features
                value = geometric_features.get(feature, 0.0)

            # Check min/max conditions
            if isinstance(condition, dict):
                if "min" in condition and value < condition["min"]:
                    return False
                if "max" in condition and value > condition["max"]:
                    return False
                if "eq" in condition and value != condition["eq"]:
                    return False
            elif isinstance(condition, (int, float)):
                if value < condition:
                    return False

        return True


# Rule type mapping for deserialization
RULE_TYPE_MAP = {
    KnowledgeCategory.MATERIAL: MaterialRule,
    KnowledgeCategory.PRECISION: PrecisionRule,
    KnowledgeCategory.STANDARD: StandardRule,
    KnowledgeCategory.FUNCTIONAL: FunctionalFeatureRule,
    KnowledgeCategory.ASSEMBLY: AssemblyRule,
    KnowledgeCategory.MANUFACTURING: ManufacturingRule,
    KnowledgeCategory.GEOMETRY: GeometryPattern,
    KnowledgeCategory.PART_TYPE: KnowledgeEntry,
}


def create_rule_from_dict(data: Dict[str, Any]) -> KnowledgeEntry:
    """Factory function to create the appropriate rule type from dict."""
    category = data.get("category", "part_type")
    if isinstance(category, str):
        category = KnowledgeCategory(category)

    rule_class = RULE_TYPE_MAP.get(category, KnowledgeEntry)
    return rule_class.from_dict(data)
