"""CAD analysis routines leveraging extracted features.

Phase 1: rule-based heuristics replacing previous static stubs.
Phase 2: ML-based classification for part type recognition.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from src.core.process_rules import recommend as recommend_process_rules
from src.models.cad_document import CadDocument

logger = logging.getLogger(__name__)

# ML分类器延迟加载
_ml_classifier: Optional[Any] = None
_ml_classifier_loaded: bool = False


def _get_ml_classifier() -> Optional[Any]:
    """延迟加载ML分类器"""
    global _ml_classifier, _ml_classifier_loaded
    if _ml_classifier_loaded:
        return _ml_classifier

    _ml_classifier_loaded = True
    try:
        from src.ml.part_classifier import PartClassifier
        model_path = os.getenv("CAD_CLASSIFIER_MODEL", "models/cad_classifier_v2.pt")
        if os.path.exists(model_path):
            _ml_classifier = PartClassifier(model_path)
            logger.info(f"ML分类器加载成功: {model_path}")
        else:
            logger.warning(f"ML分类器模型不存在: {model_path}，将使用规则分类")
    except Exception as e:
        logger.warning(f"ML分类器加载失败: {e}，将使用规则分类")
    return _ml_classifier


class CADAnalyzer:
    async def classify_part(
        self, doc: CadDocument, features: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """分类部件类型 - 优先使用ML模型，回退到规则"""
        # 尝试使用ML分类器
        ml_result = await self._classify_with_ml(doc)
        if ml_result:
            return ml_result

        # 回退到规则分类
        return await self._classify_with_rules(doc, features)

    async def _classify_with_ml(
        self, doc: CadDocument
    ) -> Optional[Dict[str, Any]]:
        """使用ML模型分类"""
        classifier = _get_ml_classifier()
        if classifier is None:
            return None

        # 获取DXF文件路径
        file_path = getattr(doc, 'file_path', None) or getattr(doc, 'source_path', None)
        if not file_path:
            return None

        try:
            result = classifier.predict(str(file_path))
            if result is None:
                return None

            return {
                "type": result.category,
                "confidence": round(result.confidence, 4),
                "probabilities": {k: round(v, 4) for k, v in result.probabilities.items()},
                "characteristics": [
                    f"ml_category:{result.category}",
                    f"ml_confidence:{result.confidence:.2%}",
                ],
                "classifier": "ml_v2",
                "rule_version": os.getenv("CLASSIFICATION_RULE_VERSION", "v2"),
            }
        except Exception as e:
            logger.warning(f"ML分类失败: {e}")
            return None

    async def _classify_with_rules(
        self, doc: CadDocument, features: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """使用规则分类（回退方案）"""
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
            "classifier": "rule_based",
            "rule_version": os.getenv("CLASSIFICATION_RULE_VERSION", "v1"),
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
