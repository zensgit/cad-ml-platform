"""
Fusion Contracts.

Defines the data structures and schemas for the Multi-Level Feature Fusion (L1-L4).
Ensures strict typing, versioning, and explainability for manufacturing decisions.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# Schema Version Control
FUSION_SCHEMA_VERSION = "v1.0"


class DecisionSource(str, Enum):
    RULE_BASED = "rule_based"  # L1/L2/L3
    AI_MODEL = "ai_model"      # L4
    HYBRID = "hybrid"          # Fusion


class ConflictLevel(str, Enum):
    NONE = "none"
    LOW = "low"      # Minor discrepancy, trust AI or Rule based on confidence
    HIGH = "high"    # Major contradiction (e.g. AI says 'Slot' but aspect ratio is 1:1)


class FusionDecision(BaseModel):
    """
    The unified output of the FusionAnalyzer.
    """
    # Core Decision
    primary_label: str = Field(description="The final determined category/feature type")
    confidence: float = Field(description="Fused confidence score (0.0 - 1.0)")
    source: DecisionSource = Field(description="Which mechanism dominated the decision")
    
    # Explainability
    reasons: List[str] = Field(default_factory=list, description="Human-readable explanation of the decision logic")
    rule_hits: List[str] = Field(default_factory=list, description="IDs of hard rules that were triggered")
    ai_raw_score: Optional[float] = Field(default=None, description="Raw confidence from L4 model before fusion")
    
    # Validation
    consistency_check: ConflictLevel = Field(default=ConflictLevel.NONE, description="Result of cross-checking AI vs Rules")
    consistency_notes: Optional[str] = Field(default=None, description="Details if conflict detected")
    
    # Metadata
    schema_version: str = Field(default=FUSION_SCHEMA_VERSION, description="Version of the fusion logic")
    feature_vector_id: Optional[str] = Field(default=None, description="ID of the combined feature vector used")


class FeatureNormalizationSchema(BaseModel):
    """
    Defines how L2/L3 features should be normalized before fusion.
    """
    keys: List[str] = Field(description="List of feature keys in order")
    scale_factors: Dict[str, float] = Field(description="Divisor for each feature (simple scaling)")
    
    def normalize(self, features: Dict[str, Any]) -> List[float]:
        vec = []
        for k in self.keys:
            val = float(features.get(k, 0.0))
            scale = self.scale_factors.get(k, 1.0)
            # Avoid division by zero
            vec.append(val / scale if scale != 0 else val)
        return vec

# Default Normalization Schema for MVP
DEFAULT_NORM_SCHEMA = FeatureNormalizationSchema(
    keys=["aspect_ratio", "complexity_score", "hole_count"],
    scale_factors={
        "aspect_ratio": 10.0,       # Expect ratios 1-10
        "complexity_score": 100.0,  # Expect scores 0-100
        "hole_count": 20.0          # Expect 0-20 holes
    }
)
