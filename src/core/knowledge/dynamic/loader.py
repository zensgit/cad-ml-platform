"""
Dynamic Knowledge Base Loader.

Provides a unified interface that combines static (builtin) knowledge
with dynamic (user-configurable) knowledge for part classification.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from src.core.knowledge.dynamic.manager import KnowledgeManager, get_knowledge_manager
from src.core.knowledge.dynamic.models import (
    KnowledgeCategory,
    KnowledgeEntry,
)

logger = logging.getLogger(__name__)


class DynamicKnowledgeBase:
    """
    Dynamic knowledge base that combines static and dynamic rules.

    This class provides:
    - Unified interface for classification hints
    - Fallback to static knowledge when dynamic is empty
    - Hot-reload support via KnowledgeManager
    - Geometry-based inference
    """

    def __init__(
        self,
        manager: Optional[KnowledgeManager] = None,
        use_static_fallback: bool = True,
    ):
        """Initialize the dynamic knowledge base.

        Args:
            manager: KnowledgeManager instance (uses global if None)
            use_static_fallback: Whether to use static knowledge as fallback
        """
        self._manager = manager or get_knowledge_manager()
        self._use_static_fallback = use_static_fallback

        # Lazy-loaded static knowledge modules
        self._static_material_kb = None
        self._static_precision_kb = None
        self._static_standards_kb = None
        self._static_functional_kb = None
        self._static_assembly_kb = None
        self._static_manufacturing_kb = None

    def _get_static_material_kb(self) -> Any:
        """Lazy load static material knowledge."""
        if self._static_material_kb is None:
            from src.core.knowledge.material_knowledge import (  # type: ignore[import-not-found]  # noqa: E501
                MaterialKnowledgeBase,
            )
            self._static_material_kb = MaterialKnowledgeBase()
        return self._static_material_kb

    def _get_static_precision_kb(self) -> Any:
        """Lazy load static precision knowledge."""
        if self._static_precision_kb is None:
            from src.core.knowledge.precision_knowledge import (  # type: ignore[import-not-found]  # noqa: E501
                PrecisionKnowledgeBase,
            )
            self._static_precision_kb = PrecisionKnowledgeBase()
        return self._static_precision_kb

    def _get_static_standards_kb(self) -> Any:
        """Lazy load static standards knowledge."""
        if self._static_standards_kb is None:
            from src.core.knowledge.standards_knowledge import (  # type: ignore[import-not-found]  # noqa: E501
                StandardsKnowledgeBase,
            )
            self._static_standards_kb = StandardsKnowledgeBase()
        return self._static_standards_kb

    def _get_static_functional_kb(self) -> Any:
        """Lazy load static functional knowledge."""
        if self._static_functional_kb is None:
            from src.core.knowledge.functional_knowledge import (  # type: ignore[import-not-found]  # noqa: E501
                FunctionalKnowledgeBase,
            )
            self._static_functional_kb = FunctionalKnowledgeBase()
        return self._static_functional_kb

    def _get_static_assembly_kb(self) -> Any:
        """Lazy load static assembly knowledge."""
        if self._static_assembly_kb is None:
            from src.core.knowledge.assembly_knowledge import (  # type: ignore[import-not-found]  # noqa: E501
                AssemblyKnowledgeBase,
            )
            self._static_assembly_kb = AssemblyKnowledgeBase()
        return self._static_assembly_kb

    def _get_static_manufacturing_kb(self) -> Any:
        """Lazy load static manufacturing knowledge."""
        if self._static_manufacturing_kb is None:
            from src.core.knowledge.manufacturing_knowledge import (  # type: ignore[import-not-found]  # noqa: E501
                ManufacturingKnowledgeBase,
            )
            self._static_manufacturing_kb = ManufacturingKnowledgeBase()
        return self._static_manufacturing_kb

    def _has_dynamic_rules(self, category: KnowledgeCategory) -> bool:
        """Check if there are dynamic rules for a category."""
        rules = self._manager.get_rules_by_category(category)
        return len(rules) > 0

    def _get_dynamic_hints(
        self,
        category: KnowledgeCategory,
        text: str,
        geometric_features: Optional[Dict[str, float]] = None,
        entity_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """Get hints from dynamic rules for a category.

        Args:
            category: Knowledge category
            text: OCR text to analyze
            geometric_features: Geometric features
            entity_counts: Entity counts

        Returns:
            Dict mapping part types to scores
        """
        hints: Dict[str, float] = {}

        rules = self._manager.get_rules_by_category(category)
        text_lower = text.lower()

        for rule in rules:
            if not rule.enabled:
                continue

            matched = False

            # Check keywords
            for keyword in rule.keywords:
                if keyword.lower() in text_lower:
                    matched = True
                    break

            # Check patterns
            if not matched:
                for pattern in rule.ocr_patterns:
                    try:
                        if re.search(pattern, text, re.IGNORECASE):
                            matched = True
                            break
                    except re.error:
                        pass

            # Apply hints if matched
            if matched:
                for part, score in rule.part_hints.items():
                    hints[part] = hints.get(part, 0) + score

        # Add geometry pattern matches
        if geometric_features or entity_counts:
            geo_patterns = self._manager.match_geometry(
                geometric_features or {},
                entity_counts or {},
            )
            for geo_pattern in geo_patterns:
                for part, score in geo_pattern.part_hints.items():
                    hints[part] = hints.get(part, 0) + score

        # Normalize
        for part in hints:
            hints[part] = min(hints[part], 1.0)

        return hints

    # ==================== Category-specific methods ====================

    def get_material_hints(
        self,
        ocr_data: Dict[str, Any],
        geometric_features: Optional[Dict[str, float]] = None,
        entity_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """Get material-based part hints.

        Args:
            ocr_data: OCR extraction results
            geometric_features: Geometric features
            entity_counts: Entity counts

        Returns:
            Dict mapping part types to confidence scores
        """
        # Extract text from OCR data
        text = self._extract_text(ocr_data)

        # Try dynamic rules first
        if self._has_dynamic_rules(KnowledgeCategory.MATERIAL):
            return self._get_dynamic_hints(
                KnowledgeCategory.MATERIAL,
                text,
                geometric_features,
                entity_counts,
            )

        # Fall back to static knowledge
        if self._use_static_fallback:
            kb = self._get_static_material_kb()
            result: Dict[str, float] = kb.get_material_hints(
                ocr_data, geometric_features, entity_counts
            )
            return result

        return {}

    def get_precision_hints(
        self,
        ocr_data: Dict[str, Any],
        geometric_features: Optional[Dict[str, float]] = None,
        entity_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """Get precision-based part hints."""
        text = self._extract_text(ocr_data)

        if self._has_dynamic_rules(KnowledgeCategory.PRECISION):
            return self._get_dynamic_hints(
                KnowledgeCategory.PRECISION,
                text,
                geometric_features,
                entity_counts,
            )

        if self._use_static_fallback:
            kb = self._get_static_precision_kb()
            result: Dict[str, float] = kb.get_precision_hints(
                ocr_data, geometric_features, entity_counts
            )
            return result

        return {}

    def get_standard_hints(
        self,
        ocr_data: Dict[str, Any],
        geometric_features: Optional[Dict[str, float]] = None,
        entity_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """Get standard-based part hints."""
        text = self._extract_text(ocr_data)

        if self._has_dynamic_rules(KnowledgeCategory.STANDARD):
            return self._get_dynamic_hints(
                KnowledgeCategory.STANDARD,
                text,
                geometric_features,
                entity_counts,
            )

        if self._use_static_fallback:
            kb = self._get_static_standards_kb()
            result: Dict[str, float] = kb.get_standard_hints(
                ocr_data, geometric_features, entity_counts
            )
            return result

        return {}

    def get_functional_hints(
        self,
        ocr_data: Dict[str, Any],
        geometric_features: Optional[Dict[str, float]] = None,
        entity_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """Get functional feature-based part hints."""
        text = self._extract_text(ocr_data)

        if self._has_dynamic_rules(KnowledgeCategory.FUNCTIONAL):
            return self._get_dynamic_hints(
                KnowledgeCategory.FUNCTIONAL,
                text,
                geometric_features,
                entity_counts,
            )

        if self._use_static_fallback:
            kb = self._get_static_functional_kb()
            result: Dict[str, float] = kb.get_feature_hints(
                ocr_data, geometric_features, entity_counts
            )
            return result

        return {}

    def get_assembly_hints(
        self,
        ocr_data: Dict[str, Any],
        geometric_features: Optional[Dict[str, float]] = None,
        entity_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """Get assembly-based part hints."""
        text = self._extract_text(ocr_data)

        if self._has_dynamic_rules(KnowledgeCategory.ASSEMBLY):
            return self._get_dynamic_hints(
                KnowledgeCategory.ASSEMBLY,
                text,
                geometric_features,
                entity_counts,
            )

        if self._use_static_fallback:
            kb = self._get_static_assembly_kb()
            result: Dict[str, float] = kb.get_assembly_hints(
                ocr_data, geometric_features, entity_counts
            )
            return result

        return {}

    def get_manufacturing_hints(
        self,
        ocr_data: Dict[str, Any],
        geometric_features: Optional[Dict[str, float]] = None,
        entity_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """Get manufacturing-based part hints."""
        text = self._extract_text(ocr_data)

        if self._has_dynamic_rules(KnowledgeCategory.MANUFACTURING):
            return self._get_dynamic_hints(
                KnowledgeCategory.MANUFACTURING,
                text,
                geometric_features,
                entity_counts,
            )

        if self._use_static_fallback:
            kb = self._get_static_manufacturing_kb()
            result: Dict[str, float] = kb.get_manufacturing_hints(
                ocr_data, geometric_features, entity_counts
            )
            return result

        return {}

    def _compute_derived_features(
        self,
        geometric_features: Dict[str, float],
        entity_counts: Dict[str, int],
    ) -> Dict[str, float]:
        """Compute derived shape indicator features for classification.

        Args:
            geometric_features: Base geometric features
            entity_counts: Entity type counts

        Returns:
            Enhanced features dict including derived indicators
        """
        import math

        features = dict(geometric_features)
        total_entities = sum(entity_counts.values()) if entity_counts else 1

        # Compute entity ratios if not present
        for entity_type in ["CIRCLE", "LINE", "ARC", "ELLIPSE", "SPLINE", "POLYLINE"]:
            ratio_key = f"{entity_type.lower()}_ratio"
            if ratio_key not in features:
                features[ratio_key] = entity_counts.get(entity_type, 0) / total_entities

        # Compute entity counts if not present
        for entity_type, count in entity_counts.items():
            count_key = f"{entity_type.lower()}_count"
            if count_key not in features:
                features[count_key] = float(count)

        # Total entities
        if "total_entities" not in features:
            features["total_entities"] = float(total_entities)

        # Curved vs straight ratios
        if "curved_ratio" not in features:
            curved = (
                entity_counts.get("CIRCLE", 0) +
                entity_counts.get("ARC", 0) +
                entity_counts.get("ELLIPSE", 0) +
                entity_counts.get("SPLINE", 0)
            )
            features["curved_ratio"] = curved / total_entities

        if "straight_ratio" not in features:
            straight = entity_counts.get("LINE", 0) + entity_counts.get("POLYLINE", 0)
            features["straight_ratio"] = straight / total_entities

        # Get base features with defaults
        sphericity = features.get("sphericity", 0.5)
        aspect_variance = features.get("aspect_variance", 0.0)
        circle_ratio = features.get("circle_ratio", 0.0)
        arc_ratio = features.get("arc_ratio", 0.0)
        complexity_score = features.get("complexity_score", 0.0)
        circle_count = features.get("circle_count", 0.0)

        # Compute derived shape type indicators
        # Rotational symmetry indicator
        if "rotational_symmetry" not in features:
            features["rotational_symmetry"] = (
                circle_ratio * (1.0 - min(aspect_variance, 1.0))
            )

        # Cylindrical shape indicator (shaft-like)
        if "cylindrical_indicator" not in features:
            features["cylindrical_indicator"] = (
                sphericity * aspect_variance
                if aspect_variance > 0.15 else 0.0
            )

        # Disk shape indicator (bearing/gear-like)
        if "disk_indicator" not in features:
            features["disk_indicator"] = (
                sphericity * (1.0 - aspect_variance)
                if sphericity > 0.5 and aspect_variance < 0.2 else 0.0
            )

        # Tooth pattern indicator (gear-like)
        if "tooth_pattern" not in features:
            features["tooth_pattern"] = (
                arc_ratio * circle_ratio
                if arc_ratio > 0.15 else 0.0
            )

        # Multi-hole pattern indicator (flange-like)
        if "multi_hole_pattern" not in features:
            features["multi_hole_pattern"] = (
                circle_ratio * math.log1p(circle_count)
                if circle_count > 5 else 0.0
            )

        # Box/housing indicator
        if "box_indicator" not in features:
            features["box_indicator"] = (
                (1.0 - sphericity) * complexity_score
                if complexity_score > 0.3 else 0.0
            )

        # Elongation and flatness (from bounding box ratios if available)
        if "elongation" not in features:
            dim_ratio_21 = features.get("dim_ratio_21", 1.0)
            dim_ratio_32 = features.get("dim_ratio_32", 1.0)
            if dim_ratio_21 > 0 and dim_ratio_32 > 0:
                product = dim_ratio_21 * dim_ratio_32
                features["elongation"] = 1.0 / product if product > 0 else 1.0
            else:
                features["elongation"] = 1.0

        if "flatness" not in features:
            dim_ratio_32 = features.get("dim_ratio_32", 1.0)
            features["flatness"] = 1.0 / dim_ratio_32 if dim_ratio_32 > 0 else 1.0

        return features

    def get_geometry_hints(
        self,
        geometric_features: Optional[Dict[str, float]] = None,
        entity_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """Get geometry pattern-based part hints.

        This matches geometric features against configured geometry patterns
        (e.g., high sphericity + low aspect variance → bearing).

        Args:
            geometric_features: Geometric features from CAD analysis
            entity_counts: Entity counts from CAD analysis

        Returns:
            Dict mapping part types to confidence scores
        """
        hints: Dict[str, float] = {}

        if not geometric_features and not entity_counts:
            return hints

        # Compute derived features for enhanced matching
        enhanced_features = self._compute_derived_features(
            geometric_features or {},
            entity_counts or {},
        )

        # Match geometry patterns from dynamic knowledge base
        geo_patterns = self._manager.match_geometry(
            enhanced_features,
            entity_counts or {},
        )

        for pattern in geo_patterns:
            for part, score in pattern.part_hints.items():
                hints[part] = hints.get(part, 0) + score

        # Normalize scores to max 1.0
        for part in hints:
            hints[part] = min(hints[part], 1.0)

        return hints

    def get_all_hints(
        self,
        ocr_data: Dict[str, Any],
        geometric_features: Optional[Dict[str, float]] = None,
        entity_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Get hints from all knowledge categories.

        Returns:
            Dict mapping category names to their hint dicts
        """
        return {
            "material": self.get_material_hints(ocr_data, geometric_features, entity_counts),
            "precision": self.get_precision_hints(ocr_data, geometric_features, entity_counts),
            "standard": self.get_standard_hints(ocr_data, geometric_features, entity_counts),
            "functional": self.get_functional_hints(ocr_data, geometric_features, entity_counts),
            "assembly": self.get_assembly_hints(ocr_data, geometric_features, entity_counts),
            "manufacturing": self.get_manufacturing_hints(
                ocr_data, geometric_features, entity_counts
            ),
            "geometry": self.get_geometry_hints(geometric_features, entity_counts),
        }

    def _extract_text(self, ocr_data: Dict[str, Any]) -> str:
        """Extract all text from OCR data."""
        texts = []

        # Title block
        title_block = ocr_data.get("title_block", {})
        if title_block:
            texts.append(str(title_block.get("part_name", "")))
            texts.append(str(title_block.get("notes", "")))
            texts.append(str(title_block.get("material", "")))

        # Text field
        if "text" in ocr_data:
            texts.append(str(ocr_data["text"]))

        # Dimensions
        for dim in ocr_data.get("dimensions", []):
            texts.append(str(dim.get("text", "")))
            texts.append(str(dim.get("tolerance", "")))

        # Notes
        for note in ocr_data.get("notes", []):
            if isinstance(note, dict):
                texts.append(str(note.get("text", "")))
            else:
                texts.append(str(note))

        # Annotations
        for ann in ocr_data.get("annotations", []):
            if isinstance(ann, dict):
                texts.append(str(ann.get("text", "")))

        return " ".join(filter(None, texts))

    # ==================== Management methods ====================

    def reload(self) -> None:
        """Reload knowledge from storage."""
        self._manager.reload()

    def get_version(self) -> str:
        """Get current knowledge version."""
        return self._manager.get_version()

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        stats = self._manager.get_stats()
        stats["static_fallback_enabled"] = self._use_static_fallback
        return stats

    def add_rule(self, rule: KnowledgeEntry) -> str:
        """Add a new knowledge rule."""
        return self._manager.add_rule(rule)

    def update_rule(self, rule: KnowledgeEntry) -> str:
        """Update an existing rule."""
        return self._manager.update_rule(rule)

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule by ID."""
        return self._manager.delete_rule(rule_id)

    def search_rules(
        self,
        query: str,
        category: Optional[KnowledgeCategory] = None,
    ) -> List[KnowledgeEntry]:
        """Search knowledge rules."""
        return self._manager.search_rules(query, category)

    def export_knowledge(self) -> Dict[str, Any]:
        """Export all dynamic knowledge."""
        return self._manager.export_knowledge()

    def import_knowledge(self, data: Dict[str, Any], merge: bool = True) -> int:
        """Import knowledge from dictionary."""
        return self._manager.import_knowledge(data, merge)
