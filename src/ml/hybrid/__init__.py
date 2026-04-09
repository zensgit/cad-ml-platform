"""
HybridClassifier enhancement modules.

Provides:
- Multi-source fusion strategies
- Confidence calibration
- Explainability
"""

from src.ml.hybrid.fusion import (
    FusionStrategy,
    SourcePrediction,
    FusionResult,
    FusionEngine,
    WeightedAverageFusion,
    VotingFusion,
    DempsterShaferFusion,
    AttentionFusion,
    MultiSourceFusion,
)

from src.ml.hybrid.calibration import (
    CalibrationMethod,
    CalibrationMetrics,
    Calibrator,
    PlattScaling,
    IsotonicCalibration,
    TemperatureScaling,
    HistogramBinning,
    BetaCalibration,
    ConfidenceCalibrator,
)

from src.ml.hybrid.explainer import (
    ExplanationType,
    FeatureContribution,
    DecisionStep,
    Counterfactual,
    Explanation,
    HybridExplainer,
)

from src.ml.hybrid.intelligence import (
    EnsembleUncertainty,
    DisagreementReport,
    CrossValidationResult,
    CalibratedConfidence,
    HybridIntelligence,
)

__all__ = [
    # Fusion
    "FusionStrategy",
    "SourcePrediction",
    "FusionResult",
    "FusionEngine",
    "WeightedAverageFusion",
    "VotingFusion",
    "DempsterShaferFusion",
    "AttentionFusion",
    "MultiSourceFusion",
    # Calibration
    "CalibrationMethod",
    "CalibrationMetrics",
    "Calibrator",
    "PlattScaling",
    "IsotonicCalibration",
    "TemperatureScaling",
    "HistogramBinning",
    "BetaCalibration",
    "ConfidenceCalibrator",
    # Explainer
    "ExplanationType",
    "FeatureContribution",
    "DecisionStep",
    "Counterfactual",
    "Explanation",
    "HybridExplainer",
    # Intelligence
    "EnsembleUncertainty",
    "DisagreementReport",
    "CrossValidationResult",
    "CalibratedConfidence",
    "HybridIntelligence",
]
