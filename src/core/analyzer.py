"""CAD analysis routines leveraging extracted features.

Phase 1: rule-based heuristics replacing previous static stubs.
Phase 2: ML-based classification for part type recognition.
Phase 3: V16 super ensemble classifier (99.65% accuracy) with speed modes.
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

# V16分类器延迟加载
_v16_classifier: Optional[Any] = None
_v16_classifier_loaded: bool = False


def _get_v16_classifier(speed_mode: str = "fast") -> Optional[Any]:
    """延迟加载V16分类器（默认使用fast模式平衡速度和精度）"""
    global _v16_classifier, _v16_classifier_loaded

    # 检查环境变量是否禁用V16
    if os.getenv("DISABLE_V16_CLASSIFIER", "").lower() in ("1", "true", "yes"):
        return None

    if _v16_classifier_loaded:
        return _v16_classifier

    _v16_classifier_loaded = True
    try:
        from src.ml.part_classifier import PartClassifierV16

        # 检查V16模型文件是否存在
        v6_path = "models/cad_classifier_v6.pt"
        v14_path = "models/cad_classifier_v14_ensemble.pt"

        if os.path.exists(v6_path) and os.path.exists(v14_path):
            env_speed_mode = os.getenv("V16_SPEED_MODE", speed_mode)
            _v16_classifier = PartClassifierV16(
                speed_mode=env_speed_mode,
                enable_cache=True,
                cache_size=int(os.getenv("V16_CACHE_SIZE", "1000")),
            )
            logger.info(f"V16分类器加载成功 (speed_mode={env_speed_mode}, 99.65%准确率)")
        else:
            logger.info("V16模型文件不完整，将回退到V6分类器")
    except Exception as e:
        logger.warning(f"V16分类器加载失败: {e}，将回退到V6分类器")

    return _v16_classifier


def _get_ml_classifier() -> Optional[Any]:
    """延迟加载ML分类器（V6回退方案）"""
    global _ml_classifier, _ml_classifier_loaded
    if _ml_classifier_loaded:
        return _ml_classifier

    _ml_classifier_loaded = True
    try:
        from src.ml.part_classifier import PartClassifier
        model_path = os.getenv("CAD_CLASSIFIER_MODEL", "models/cad_classifier_v6.pt")
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
        """分类部件类型 - 优先使用V16，回退到V6，最后规则"""
        # 尝试使用V16分类器（最高精度）
        v16_result = await self._classify_with_v16(doc)
        if v16_result:
            return v16_result

        # 回退到V6分类器
        ml_result = await self._classify_with_ml(doc)
        if ml_result:
            return ml_result

        # 回退到规则分类
        return await self._classify_with_rules(doc, features)

    async def _classify_with_v16(
        self, doc: CadDocument
    ) -> Optional[Dict[str, Any]]:
        """使用V16超级集成分类器（99.65%准确率）"""
        # Avoid loading heavyweight models when we don't have an on-disk file path.
        # The analyze endpoint typically operates on uploaded bytes, so `CadDocument`
        # does not carry a stable file path unless explicitly provided by an adapter.
        file_path = getattr(doc, "file_path", None) or getattr(doc, "source_path", None)
        if not file_path:
            return None

        classifier = _get_v16_classifier()
        if classifier is None:
            return None

        try:
            result = classifier.predict(str(file_path))
            if result is None:
                return None

            response = {
                "type": result.category,
                "confidence": round(result.confidence, 4),
                "probabilities": {k: round(v, 4) for k, v in result.probabilities.items()},
                "characteristics": [
                    f"ml_category:{result.category}",
                    f"ml_confidence:{result.confidence:.2%}",
                ],
                "classifier": result.model_version,
                "rule_version": os.getenv("CLASSIFICATION_RULE_VERSION", "v16"),
            }

            # 添加V16特有字段
            if result.needs_review:
                response["needs_review"] = True
                response["review_reason"] = result.review_reason
            if result.top2_category:
                response["top2_category"] = result.top2_category
                response["top2_confidence"] = round(result.top2_confidence, 4)

            return response
        except Exception as e:
            logger.warning(f"V16分类失败: {e}，将回退到V6")
            return None

    async def _classify_with_ml(
        self, doc: CadDocument
    ) -> Optional[Dict[str, Any]]:
        """使用ML模型分类（V6回退方案）"""
        file_path = getattr(doc, "file_path", None) or getattr(doc, "source_path", None)
        if not file_path:
            return None

        classifier = _get_ml_classifier()
        if classifier is None:
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
                "classifier": "ml_v6",
                "rule_version": os.getenv("CLASSIFICATION_RULE_VERSION", "v6"),
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
