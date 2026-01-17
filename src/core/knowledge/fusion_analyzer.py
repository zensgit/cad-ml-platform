"""
Fusion Analyzer.

The central "brain" that orchestrates L1-L4 feature inputs to make a reliable manufacturing decision.
Implements the "Ensemble & Fallback" strategy with strict guardrails.
"""

import logging
from typing import Any, Dict, Optional

from src.core.knowledge.fusion_contracts import (
    FusionDecision,
    DecisionSource,
    ConflictLevel,
    DEFAULT_NORM_SCHEMA,
    FUSION_SCHEMA_VERSION
)
from src.core.errors_extended import ErrorCode, create_extended_error

logger = logging.getLogger(__name__)

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
        
        ai_label = l4_prediction.get("label") if l4_prediction else None
        ai_conf = l4_prediction.get("confidence", 0.0) if l4_prediction else 0.0
        
        if ai_label == "Slot" and not is_geometric_slot:
            conflict = ConflictLevel.HIGH
            conflict_note = f"AI detected Slot but Aspect Ratio is {aspect_ratio:.2f} (Expected > 1.5)"
            reasons.append("L2/L4 Conflict: Geometry does not support AI prediction")
        
        # --- Step 3: Decision Logic ---
        
        # Case A: Strong AI Confidence + No Conflict -> Trust AI
        if ai_conf >= self.ai_threshold and conflict != ConflictLevel.HIGH:
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

    def _fallback_decision(self, label: str, conf: float, reasons: list, source: DecisionSource) -> FusionDecision:
        return FusionDecision(
            primary_label=label,
            confidence=conf,
            source=source,
            reasons=reasons,
            schema_version=FUSION_SCHEMA_VERSION
        )

# Singleton
_fusion = FusionAnalyzer()

def get_fusion_analyzer():
    return _fusion
