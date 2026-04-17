"""Vector registration and similarity dispatch helpers for analyze flows."""

from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional

from src.core.classification.vector_metadata import build_vector_registration_metadata
from src.core.feature_extractor import FeatureExtractor
from src.core.similarity import FaissVectorStore, compute_similarity, has_vector, register_vector
from src.core.vector_layouts import VECTOR_LAYOUT_BASE, VECTOR_LAYOUT_L3

logger = logging.getLogger(__name__)

MetricObserver = Callable[[Any], Any]
MetadataUpdater = Callable[[str, Dict[str, str]], Any]
QdrantStoreGetter = Callable[[], Any]
QdrantSimilarityFn = Callable[[Any, str, List[float]], Awaitable[Dict[str, Any]]]
LocalRegisterFn = Callable[[str, List[float], Dict[str, str]], bool]
LocalSimilarityFn = Callable[[str, List[float]], Dict[str, Any]]
HasVectorFn = Callable[[str], bool]
FeatureExtractorFactory = Callable[[], Any]
MetadataBuilder = Callable[..., Dict[str, str]]
FaissStoreFactory = Callable[[], Any]


def _default_feature_extractor_factory() -> Any:
    return FeatureExtractor()


def _default_faiss_store_factory() -> Any:
    return FaissVectorStore()


def _default_memory_meta_updater(vector_id: str, metadata: Dict[str, str]) -> None:
    try:
        vector_meta = __import__(
            "src.core.similarity", fromlist=["_VECTOR_META"]
        )._VECTOR_META  # type: ignore[attr-defined]
        vector_meta[vector_id].update(metadata)
    except Exception:
        pass


def _build_feature_vector(
    *,
    features: Mapping[str, Any],
    features_3d: Optional[Mapping[str, Any]],
    feature_extractor_factory: Optional[FeatureExtractorFactory],
) -> Dict[str, Any]:
    extractor = (feature_extractor_factory or _default_feature_extractor_factory)()
    feature_vector: List[float] = extractor.flatten(features)
    vector_layout = VECTOR_LAYOUT_BASE
    l3_dim: Optional[int] = None

    embedding_vector = None
    if isinstance(features_3d, Mapping):
        embedding_vector = features_3d.get("embedding_vector")
    if embedding_vector is not None:
        l3_dim = len(embedding_vector)
        feature_vector.extend(float(x) for x in embedding_vector)
        vector_layout = VECTOR_LAYOUT_L3

    return {
        "feature_vector": feature_vector,
        "vector_layout": vector_layout,
        "l3_dim": l3_dim,
    }


async def run_vector_pipeline(
    *,
    analysis_id: str,
    doc: Any,
    features: Mapping[str, Any],
    features_3d: Optional[Mapping[str, Any]],
    material: Optional[str],
    classification_meta: Optional[Mapping[str, Any]] = None,
    calculate_similarity: bool = False,
    reference_id: Optional[str] = None,
    feature_version: Optional[str] = None,
    get_qdrant_store: Optional[QdrantStoreGetter] = None,
    compute_qdrant_similarity: Optional[QdrantSimilarityFn] = None,
    feature_extractor_factory: Optional[FeatureExtractorFactory] = None,
    metadata_builder: Optional[MetadataBuilder] = None,
    register_local_vector: LocalRegisterFn = register_vector,
    compute_local_similarity: LocalSimilarityFn = compute_similarity,
    has_local_vector: HasVectorFn = has_vector,
    faiss_store_factory: Optional[FaissStoreFactory] = None,
    vector_material_observer: Optional[MetricObserver] = None,
    feature_dimension_observer: Optional[MetricObserver] = None,
    memory_meta_updater: Optional[MetadataUpdater] = None,
    logger_instance: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Register an analysis vector and optionally compute similarity."""
    active_logger = logger_instance or logger
    similarity_payload: Optional[Dict[str, Any]] = None
    feature_vector: List[float] = []
    feature_vector_ready = False
    metadata: Dict[str, str] = {}
    registered = False
    qdrant_store = get_qdrant_store() if get_qdrant_store is not None else None

    try:
        vector_context = _build_feature_vector(
            features=features,
            features_3d=features_3d,
            feature_extractor_factory=feature_extractor_factory,
        )
        feature_vector = vector_context["feature_vector"]
        feature_vector_ready = True
        vector_layout = vector_context["vector_layout"]
        l3_dim = vector_context["l3_dim"]
        material_used = material or "unknown"
        metadata = (metadata_builder or build_vector_registration_metadata)(
            material=material_used,
            doc=doc,
            features=features,
            feature_vector=feature_vector,
            feature_version=feature_version or os.getenv("FEATURE_VERSION", "v1"),
            vector_layout=vector_layout,
            classification_meta=classification_meta or {},
            l3_dim=l3_dim,
        )

        stored_in_qdrant = False
        if qdrant_store is not None:
            await qdrant_store.register_vector(
                analysis_id,
                feature_vector,
                metadata=metadata,
            )
            registered = True
            stored_in_qdrant = True
        else:
            registered = register_local_vector(
                analysis_id,
                feature_vector,
                metadata,
            )

        if registered:
            if vector_material_observer is not None:
                vector_material_observer(material_used)
            if os.getenv("VECTOR_STORE_BACKEND", "memory") == "faiss":
                try:
                    (faiss_store_factory or _default_faiss_store_factory)().add(
                        analysis_id, feature_vector
                    )
                except Exception:
                    pass
            if feature_dimension_observer is not None:
                feature_dimension_observer(len(feature_vector))
            if not stored_in_qdrant:
                (memory_meta_updater or _default_memory_meta_updater)(
                    analysis_id, metadata
                )
    except Exception:
        pass

    if calculate_similarity and reference_id and feature_vector_ready:
        if qdrant_store is not None and compute_qdrant_similarity is not None:
            similarity_payload = await compute_qdrant_similarity(
                qdrant_store,
                reference_id,
                feature_vector,
            )
        else:
            similarity_payload = compute_local_similarity(reference_id, feature_vector)
    elif reference_id:
        if qdrant_store is not None:
            reference = await qdrant_store.get_vector(reference_id)
            if reference is None:
                similarity_payload = {
                    "reference_id": reference_id,
                    "status": "reference_not_found",
                }
        elif not has_local_vector(reference_id):
            similarity_payload = {
                "reference_id": reference_id,
                "status": "reference_not_found",
            }

    return {
        "registered": registered,
        "similarity": similarity_payload,
        "vector_metadata": metadata,
        "feature_vector_dim": len(feature_vector),
    }


__all__ = ["run_vector_pipeline"]
