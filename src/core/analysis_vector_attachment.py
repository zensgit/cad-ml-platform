from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, Dict, Optional


VectorPipeline = Callable[..., Awaitable[Dict[str, Any]]]
MetricObserver = Callable[[float], Any]


async def attach_analysis_vector_context(
    *,
    analysis_id: str,
    doc: Any,
    features: Dict[str, Any],
    features_3d: Dict[str, Any],
    material: Optional[str],
    classification_meta: Dict[str, Any],
    calculate_similarity: bool,
    reference_id: Optional[str],
    results: Dict[str, Any],
    stage_times: Dict[str, float],
    started_at: float,
    vector_pipeline: VectorPipeline,
    get_qdrant_store: Callable[[], Any],
    compute_qdrant_similarity: Callable[..., Any],
    vector_material_observer: Callable[[str], Any],
    feature_dimension_observer: MetricObserver,
    similarity_stage_observer: MetricObserver,
) -> Dict[str, Any]:
    vector_context = await vector_pipeline(
        analysis_id=analysis_id,
        doc=doc,
        features=features,
        features_3d=features_3d,
        material=material,
        classification_meta=classification_meta,
        calculate_similarity=calculate_similarity,
        reference_id=reference_id,
        get_qdrant_store=get_qdrant_store,
        compute_qdrant_similarity=compute_qdrant_similarity,
        vector_material_observer=vector_material_observer,
        feature_dimension_observer=feature_dimension_observer,
    )
    if vector_context["similarity"] is not None:
        results["similarity"] = vector_context["similarity"]
    if "similarity" in results:
        similarity_stage_duration = time.time() - started_at - sum(stage_times.values())
        stage_times["similarity"] = similarity_stage_duration
        similarity_stage_observer(similarity_stage_duration)
    return vector_context
