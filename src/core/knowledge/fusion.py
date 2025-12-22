"""
Multi-modal Fusion Engine (L3).

Combines signals from:
1. Knowledge Base (Rules/Expert System)
2. OCR (Text/Semantic)
3. 3D Vision (Deep Geometric Learning)
"""

import logging
from typing import Dict, Any, List, Optional
from src.core.knowledge.dynamic.manager import get_knowledge_manager

logger = logging.getLogger(__name__)

class FusionClassifier:
    """
    L3 Fusion Classifier.
    Integrates 2D, 3D, and Text signals.
    """
    
    def __init__(self):
        self.km = get_knowledge_manager()
        
    def classify(
        self, 
        text_signals: str,
        features_2d: Dict[str, Any],
        features_3d: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main entry point for multi-modal classification.
        """
        
        # 1. Get Rule-based/2D Hints (Existing L1/L2 capability)
        # We assume features_2d contains 'geometric_features' and 'entity_counts'
        hints_2d = self.km.get_part_hints(
            text=text_signals,
            geometric_features=features_2d.get("geometric_features"),
            entity_counts=features_2d.get("entity_counts")
        )
        
        # 2. Analyze 3D Signals (L3 capability)
        hints_3d = self._analyze_3d_signals(features_3d)
        
        # 3. Fuse Scores
        final_scores = self._fuse_scores(hints_2d, hints_3d)

        # 3b. Strong 3D override: if 3D signal is high-confidence, prioritize it.
        if hints_3d:
            top_3d_part, top_3d_score = max(hints_3d.items(), key=lambda kv: kv[1])
            if top_3d_score >= 0.6:
                current_max = max(final_scores.values(), default=0.0)
                final_scores[top_3d_part] = max(final_scores.get(top_3d_part, 0.0), current_max + 0.01)
        
        # 4. Determine Winner
        sorted_parts = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        best_part = sorted_parts[0][0] if sorted_parts else "unknown"
        best_score = sorted_parts[0][1] if sorted_parts else 0.0
        
        alternatives = [
            {"part_type": p, "confidence": s} 
            for p, s in sorted_parts[1:5] 
            if s > 0.1
        ]
        
        return {
            "type": best_part,
            "confidence": best_score,
            "alternatives": alternatives,
            "fusion_breakdown": {
                "2d_score": hints_2d.get(best_part, 0),
                "3d_score": hints_3d.get(best_part, 0),
                "source": "l3_fusion"
            }
        }

    def _analyze_3d_signals(self, features_3d: Dict[str, Any]) -> Dict[str, float]:
        """
        Heuristic logic based on 3D B-Rep features.
        In a future version, this would rely heavily on the UV-Net vector distance
        to known classes prototypes.
        """
        hints = {}
        
        if not features_3d.get("valid_3d"):
            return hints
            
        surfaces = features_3d.get("surface_types", {})
        total_faces = features_3d.get("faces", 1)
        
        # Example L3 Logic:
        # High percentage of Cylindrical faces -> Shaft or Bolt
        cyl_ratio = surfaces.get("cylinder", 0) / total_faces
        if cyl_ratio > 0.4:
            hints["shaft"] = 0.6
            hints["bolt"] = 0.4
            
        # High percentage of Planar faces + 6 faces -> Cube/Block
        plane_ratio = surfaces.get("plane", 0) / total_faces
        if plane_ratio > 0.8 and 5 <= total_faces <= 8:
            hints["block"] = 0.7
            hints["plate"] = 0.5

        # Torus usually means something rotational/ring-like
        if surfaces.get("torus", 0) > 0:
            hints["bearing"] = 0.5
            hints["seal"] = 0.6
            
        return hints

    def _fuse_scores(self, hints_2d: Dict[str, float], hints_3d: Dict[str, float]) -> Dict[str, float]:
        """
        Weighted fusion of signals.
        """
        # Start with 2D scores
        final = hints_2d.copy()
        
        # Add 3D scores with weight
        weight_3d = 0.8  # 3D is trustworthy if available
        
        for part, score in hints_3d.items():
            final[part] = final.get(part, 0) + (score * weight_3d)
            
        # Normalize? Not strictly necessary for ranking, but good for confidence
        # Simple clamping for now
        for part in final:
            if final[part] > 1.0: final[part] = 1.0
            
        return final

# Singleton
_fusion = FusionClassifier()

def get_fusion_classifier():
    return _fusion
