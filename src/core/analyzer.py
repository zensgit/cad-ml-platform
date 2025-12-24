"""CAD analysis routines leveraging extracted features.

Phase 1: rule-based heuristics replacing previous static stubs.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.core.process_rules import recommend as recommend_process_rules
from src.models.cad_document import CadDocument


class CADAnalyzer:
    async def classify_part(
        self, doc: CadDocument, features: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        geometric = features.get("geometric", [])
        semantic = features.get("semantic", [])
        entity_count = geometric[0] if len(geometric) > 0 else doc.entity_count()
        volume = geometric[4] if len(geometric) > 4 else doc.bounding_box.volume_estimate
        layer_count = semantic[0] if len(semantic) > 0 else len(doc.layers)
        # Simple heuristic taxonomy
        if entity_count < 20 and volume < 1e4:
            part_type = "simple_plate"
        elif entity_count < 200:
            part_type = "moderate_component"
        else:
            part_type = "complex_assembly"
        confidence = 0.6 if part_type != "complex_assembly" else 0.55
        return {
            "type": part_type,
            "confidence": confidence,
            "characteristics": [
                f"entities:{entity_count}",
                f"layers:{layer_count}",
                f"volume_estimate:{volume:.2f}",
            ],
            "rule_version": __import__("os").getenv("CLASSIFICATION_RULE_VERSION", "v1"),
        }

    async def check_quality(
        self, doc: CadDocument, features: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        geometric = features.get("geometric", [])
        semantic = features.get("semantic", [])
        entity_count = geometric[0] if geometric else doc.entity_count()
        layer_count = semantic[0] if semantic else len(doc.layers)
        issues = []
        suggestions = []
        if layer_count == 0:
            issues.append("no_layers_defined")
            suggestions.append("organize_entities_into_layers")
        if entity_count == 0:
            issues.append("empty_document")
        score = max(0.0, 1.0 - (0.05 * len(issues)))
        return {"score": round(score, 2), "issues": issues, "suggestions": suggestions}

    async def recommend_process(
        self, doc: CadDocument, features: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        geometric = features.get("geometric", [])
        volume = geometric[4] if len(geometric) > 4 else doc.bounding_box.volume_estimate
        entity_count = geometric[0] if geometric else doc.entity_count()
        complexity = doc.complexity_bucket()
        material = doc.metadata.get("material", "steel")
        rule = recommend_process_rules(material=material, complexity=complexity, volume=volume)
        return {
            "recommended_process": rule.get("primary"),
            "alternatives": rule.get("alternatives", []),
            "parameters": {
                "est_volume": volume,
                "entity_count": entity_count,
                "complexity": complexity,
                "material": material,
                "matched_volume_threshold": rule.get("matched_volume_threshold"),
            },
        }
