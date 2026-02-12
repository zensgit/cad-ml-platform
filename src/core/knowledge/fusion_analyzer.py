"""
Fusion Analyzer.

The central "brain" that orchestrates L1-L4 feature inputs to make a reliable
manufacturing decision.
Implements the "Ensemble & Fallback" strategy with strict guardrails.
"""

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

from src.core.knowledge.fusion_contracts import (
    FusionDecision,
    DecisionSource,
    ConflictLevel,
    DEFAULT_NORM_SCHEMA,
    FUSION_SCHEMA_VERSION
)
from src.models.cad_document import CadDocument

logger = logging.getLogger(__name__)


def build_doc_metadata(doc: CadDocument) -> Dict[str, Any]:
    valid_format = doc.format in {"dxf", "dwg", "step", "stp", "iges", "igs", "stl"}
    return {
        "valid_format": valid_format,
        "format": doc.format,
        "entity_count": doc.entity_count(),
    }


def build_l2_features(doc: CadDocument) -> Dict[str, Any]:
    bbox = doc.bounding_box
    width = max(bbox.width, 0.0)
    height = max(bbox.height, 0.0)
    depth = max(bbox.depth, 0.0)

    min_dim = min(d for d in (width, height) if d > 0.0) if width > 0 and height > 0 else 0.0
    max_dim = max(width, height)
    aspect_ratio = max_dim / min_dim if min_dim > 0 else 1.0
    try:
        dimension_count = int(doc.metadata.get("dimension_count") or 0)
    except (TypeError, ValueError):
        dimension_count = 0
    text_content = doc.metadata.get("text_content")
    if isinstance(text_content, list):
        text_count = len([t for t in text_content if str(t).strip()])
    else:
        text_count = 0

    return {
        "aspect_ratio": aspect_ratio,
        "complexity_score": float(doc.entity_count()),
        "hole_count": 0.0,
        "bbox_width": width,
        "bbox_height": height,
        "bbox_depth": depth,
        "entity_count": float(doc.entity_count()),
        "layer_count": float(len(doc.layers)),
        "dimension_count": float(dimension_count),
        "text_count": float(text_count),
    }


class FusionAnalyzer:
    def __init__(self, ai_confidence_threshold: float = 0.7):
        self.ai_threshold = ai_confidence_threshold
        self.norm_schema = DEFAULT_NORM_SCHEMA
        self.refresh_from_env()

    def refresh_from_env(self) -> None:
        """Reload override settings from environment variables."""
        self.graph2d_override_labels = self._load_graph2d_override_labels()
        self.graph2d_override_min_conf = self._load_graph2d_override_min_conf()
        self.graph2d_override_low_conf_labels = self._load_graph2d_override_low_conf_labels()
        self.graph2d_override_low_conf_min = self._load_graph2d_override_low_conf_min()

    def analyze(
        self,
        doc_metadata: Dict[str, Any],       # L1: Format, basic meta
        l2_features: Dict[str, Any],        # L2: 2D Projection stats
        l3_features: Dict[str, Any],        # L3: 3D Physics/B-Rep stats
        l4_prediction: Optional[Dict[str, Any]] = None  # L4: AI Output
    ) -> FusionDecision:
        """
        Execute the fusion logic pipeline.
        """
        reasons = []
        rule_hits = []
        normalized_vec, feature_vector_id = self._normalize_features(l2_features, l3_features)
        rule_confidence = self._estimate_rule_confidence(l2_features, l3_features, normalized_vec)
        
        # --- Step 1: L1 Hard Guardrails ---
        # Example: Reject if format is unknown or file is empty
        if not doc_metadata.get("valid_format", True):
            return self._fallback_decision(
                "Unknown", 0.0, ["L1: Invalid file format"], DecisionSource.RULE_BASED
            )

        # --- Step 2: Consistency Check (Rule vs AI) ---
        conflict = ConflictLevel.NONE
        conflict_note = None
        
        # Derive key indicators from L2/L3
        # Example logic: "Slot" implies Aspect Ratio > 1.5
        aspect_ratio = l2_features.get("aspect_ratio", 1.0)
        is_geometric_slot = aspect_ratio > 1.5
        
        ai_label = (
            str(l4_prediction.get("label"))
            if l4_prediction and l4_prediction.get("label") is not None
            else None
        )
        ai_conf = l4_prediction.get("confidence", 0.0) if l4_prediction else 0.0
        
        if ai_label == "Slot" and not is_geometric_slot:
            conflict = ConflictLevel.HIGH
            conflict_note = f"AI detected Slot but Aspect Ratio is {aspect_ratio:.2f} (Expected > 1.5)"
            reasons.append("L2/L4 Conflict: Geometry does not support AI prediction")
        ai_conf_calibrated = self._calibrate_ai_confidence(ai_conf, conflict, normalized_vec)

        if self._should_graph2d_override(l4_prediction, conflict, ai_conf_calibrated):
            reasons.append(
                f"Graph2D override for label {ai_label} at confidence {ai_conf_calibrated:.2f}"
            )
            rule_hits.append("GRAPH2D_OVERRIDE")
            return FusionDecision(
                primary_label=ai_label or "Unknown",
                confidence=ai_conf_calibrated,
                source=DecisionSource.HYBRID,
                reasons=reasons,
                rule_hits=rule_hits,
                ai_raw_score=ai_conf,
                consistency_check=conflict,
                consistency_notes=conflict_note,
                schema_version=FUSION_SCHEMA_VERSION,
                feature_vector_id=feature_vector_id,
            )
        
        # --- Step 3: Decision Logic ---
        
        # Case A: Strong AI Confidence + No Conflict -> Trust AI
        if (
            ai_label is not None
            and ai_conf_calibrated >= self.ai_threshold
            and conflict != ConflictLevel.HIGH
        ):
            reasons.append(
                f"AI confidence {ai_conf_calibrated:.2f} exceeds threshold (raw {ai_conf:.2f})"
            )
            return FusionDecision(
                primary_label=ai_label,
                confidence=ai_conf_calibrated,
                source=DecisionSource.AI_MODEL,
                reasons=reasons,
                ai_raw_score=ai_conf,
                consistency_check=conflict,
                consistency_notes=conflict_note,
                schema_version=FUSION_SCHEMA_VERSION,
                feature_vector_id=feature_vector_id,
            )
            
        # Case B: Conflict or Low AI Confidence -> Fallback to Rules
        # Simple heuristic rule for demo:
        if is_geometric_slot:
            primary = "Slot"
            rule_hits.append("RULE_ASPECT_RATIO_SLOT")
            conf = max(0.6, rule_confidence)
            reasons.append("Fallback to L2 Geometry: High aspect ratio implies Slot")
        else:
            primary = "Standard_Part"
            rule_hits.append("RULE_DEFAULT")
            conf = max(0.5, rule_confidence)
            reasons.append("Default fallback: No specific features detected")

        return FusionDecision(
            primary_label=primary,
            confidence=conf,
            source=DecisionSource.RULE_BASED,
            reasons=reasons,
            rule_hits=rule_hits,
            ai_raw_score=ai_conf,
            consistency_check=conflict,
            consistency_notes=conflict_note,
            schema_version=FUSION_SCHEMA_VERSION,
            feature_vector_id=feature_vector_id,
        )

    def _fallback_decision(
        self,
        label: str,
        conf: float,
        reasons: List[str],
        source: DecisionSource,
    ) -> FusionDecision:
        return FusionDecision(
            primary_label=label,
            confidence=conf,
            source=source,
            reasons=reasons,
            schema_version=FUSION_SCHEMA_VERSION,
        )

    def _flatten_l3_features(self, l3_features: Dict[str, Any]) -> Dict[str, float]:
        surfaces = l3_features.get("surface_types", {}) or {}
        return {
            "faces": float(l3_features.get("faces", 0) or 0),
            "edges": float(l3_features.get("edges", 0) or 0),
            "volume": float(l3_features.get("volume", 0) or 0),
            "surface_area": float(l3_features.get("surface_area", 0) or 0),
            "bbox_volume": float(l3_features.get("bbox_volume", 0) or 0),
            "surface_plane": float(surfaces.get("plane", 0) or 0),
            "surface_cylinder": float(surfaces.get("cylinder", 0) or 0),
            "surface_cone": float(surfaces.get("cone", 0) or 0),
            "surface_sphere": float(surfaces.get("sphere", 0) or 0),
            "surface_torus": float(surfaces.get("torus", 0) or 0),
            "surface_bspline": float(surfaces.get("bspline", 0) or 0),
        }

    def _normalize_features(
        self,
        l2_features: Dict[str, Any],
        l3_features: Dict[str, Any],
    ) -> tuple[List[float], Optional[str]]:
        merged = dict(l2_features)
        merged.update(self._flatten_l3_features(l3_features))
        vec = self.norm_schema.normalize(merged)
        if not vec:
            return vec, None
        vector_str = ",".join(f"{v:.6f}" for v in vec)
        vector_hash = hashlib.sha256(vector_str.encode("utf-8")).hexdigest()[:12]
        return vec, vector_hash

    def _calibrate_ai_confidence(
        self,
        ai_confidence: float,
        conflict: ConflictLevel,
        normalized_vec: List[float],
    ) -> float:
        conf = max(0.0, min(1.0, float(ai_confidence)))
        if conflict == ConflictLevel.HIGH:
            conf *= 0.6
        elif conflict == ConflictLevel.LOW:
            conf *= 0.85
        if normalized_vec and max(normalized_vec) < 0.02:
            conf *= 0.85
        return max(0.0, min(1.0, conf))

    def _estimate_rule_confidence(
        self,
        l2_features: Dict[str, Any],
        l3_features: Dict[str, Any],
        normalized_vec: List[float],
    ) -> float:
        conf = 0.45
        if l2_features.get("entity_count", 0) > 0:
            conf += 0.05
        if l2_features.get("dimension_count", 0) > 0:
            conf += 0.05
        if l3_features.get("faces", 0) > 0:
            conf += 0.1
        if l3_features.get("surface_area", 0) or l3_features.get("volume", 0):
            conf += 0.1
        if normalized_vec and max(normalized_vec) > 0.5:
            conf += 0.05
        return max(0.4, min(0.8, conf))

    def _load_graph2d_override_labels(self) -> set[str]:
        raw = os.getenv("FUSION_GRAPH2D_OVERRIDE_LABELS", "模板,零件图,装配图")
        return {label.strip() for label in raw.split(",") if label.strip()}

    def _load_graph2d_override_min_conf(self) -> float:
        raw = os.getenv("FUSION_GRAPH2D_OVERRIDE_MIN_CONF", "0.6")
        try:
            return self._clamp_conf(float(raw), 0.6)
        except (TypeError, ValueError):
            return 0.6

    def _load_graph2d_override_low_conf_labels(self) -> set[str]:
        raw = os.getenv("FUSION_GRAPH2D_OVERRIDE_LOW_CONF_LABELS", "机械制图,零件图")
        return {label.strip() for label in raw.split(",") if label.strip()}

    def _load_graph2d_override_low_conf_min(self) -> float:
        raw = os.getenv("FUSION_GRAPH2D_OVERRIDE_LOW_CONF_MIN_CONF", "0.6")
        try:
            return self._clamp_conf(float(raw), 0.6)
        except (TypeError, ValueError):
            return 0.6

    @staticmethod
    def _clamp_conf(value: float, default: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value if value == value else default

    def _should_graph2d_override(
        self,
        l4_prediction: Optional[Dict[str, Any]],
        conflict: ConflictLevel,
        ai_conf: float,
    ) -> bool:
        if not l4_prediction or conflict == ConflictLevel.HIGH:
            return False
        if l4_prediction.get("source") != "graph2d":
            return False
        label = str(l4_prediction.get("label") or "").strip()
        if not label or label not in self.graph2d_override_labels:
            return False
        required_min = self.graph2d_override_min_conf
        if label in self.graph2d_override_low_conf_labels:
            required_min = max(required_min, self.graph2d_override_low_conf_min)
        return float(ai_conf) >= required_min

# Singleton
_fusion: Optional[FusionAnalyzer] = None


def get_fusion_analyzer(reset: bool = False, refresh_env: bool = False) -> FusionAnalyzer:
    """Return a singleton FusionAnalyzer with optional env refresh."""
    global _fusion
    if _fusion is None or reset:
        _fusion = FusionAnalyzer()
    elif refresh_env:
        _fusion.refresh_from_env()
    return _fusion
