"""
Fusion Analyzer.

The central "brain" that orchestrates L1-L4 feature inputs to make a reliable
manufacturing decision.
Implements the "Ensemble & Fallback" strategy with strict guardrails.
"""

import logging
from typing import Any, Dict, List, Optional

from src.core.knowledge.fusion_contracts import (
    FusionDecision,
    DecisionSource,
    ConflictLevel,
    DEFAULT_NORM_SCHEMA,
    FUSION_SCHEMA_VERSION
)
from src.core.errors_extended import ErrorCode, create_extended_error
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
        
        # --- Step 3: Decision Logic ---
        
        # Case A: Strong AI Confidence + No Conflict -> Trust AI
        if ai_label is not None and ai_conf >= self.ai_threshold and conflict != ConflictLevel.HIGH:
            reasons.append(f"AI confidence {ai_conf:.2f} exceeds threshold")
            return FusionDecision(
                primary_label=ai_label,
                confidence=ai_conf,
                source=DecisionSource.AI_MODEL,
                reasons=reasons,
                ai_raw_score=ai_conf,
                consistency_check=conflict,
                consistency_notes=conflict_note,
                schema_version=FUSION_SCHEMA_VERSION
            )
            
        # Case B: Conflict or Low AI Confidence -> Fallback to Rules
        # Simple heuristic rule for demo:
        if is_geometric_slot:
            primary = "Slot"
            rule_hits.append("RULE_ASPECT_RATIO_SLOT")
            conf = 0.6  # Rule based is moderate confidence
            reasons.append("Fallback to L2 Geometry: High aspect ratio implies Slot")
        else:
            primary = "Standard_Part"
            rule_hits.append("RULE_DEFAULT")
            conf = 0.5
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
            schema_version=FUSION_SCHEMA_VERSION
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
            schema_version=FUSION_SCHEMA_VERSION
        )

# Singleton
_fusion = FusionAnalyzer()

def get_fusion_analyzer() -> FusionAnalyzer:
    return _fusion
