"""Classification helpers shared across API and core analyzers."""

from src.core.classification.active_learning_policy import (
    flag_classification_for_review,
)
from src.core.classification.baseline_policy import (
    build_baseline_classification_context,
    build_baseline_classification_payload,
)
from src.core.classification.batch_classify_pipeline import (
    build_batch_classify_item,
    run_batch_classify_pipeline,
)
from src.core.classification.coarse_labels import labels_conflict, normalize_coarse_label
from src.core.classification.classification_pipeline import (
    run_classification_pipeline,
)
from src.core.classification.decision_contract import (
    build_classification_decision_contract,
    extract_label_decision_contract,
)
from src.core.classification.decision_service import (
    DECISION_CONTRACT_VERSION,
    DecisionService,
)
from src.core.classification.finalization import finalize_classification_payload
from src.core.classification.fusion_pipeline import (
    build_fusion_classification_context,
)
from src.core.classification.hybrid_override_pipeline import (
    build_hybrid_override_context,
)
from src.core.classification.override_policy import (
    apply_fusion_override,
    apply_hybrid_override,
)
from src.core.classification.part_family import normalize_part_family_prediction
from src.core.classification.review_governance import (
    build_review_governance,
    derive_confidence_band,
)
from src.core.classification.shadow_pipeline import (
    build_shadow_classification_context,
)
from src.core.classification.vector_metadata import (
    build_vector_registration_metadata,
    extract_vector_label_contract,
)

__all__ = [
    "flag_classification_for_review",
    "build_baseline_classification_context",
    "build_baseline_classification_payload",
    "build_batch_classify_item",
    "run_batch_classify_pipeline",
    "labels_conflict",
    "normalize_coarse_label",
    "run_classification_pipeline",
    "extract_label_decision_contract",
    "build_classification_decision_contract",
    "DECISION_CONTRACT_VERSION",
    "DecisionService",
    "finalize_classification_payload",
    "build_fusion_classification_context",
    "build_hybrid_override_context",
    "apply_fusion_override",
    "apply_hybrid_override",
    "normalize_part_family_prediction",
    "build_review_governance",
    "derive_confidence_band",
    "build_shadow_classification_context",
    "build_vector_registration_metadata",
    "extract_vector_label_contract",
]
