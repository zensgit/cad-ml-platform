"""Shared feature extraction and cache pipeline for analyze flows."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

from src.core.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

FeatureExtractorFactory = Callable[[], Any]
FeatureCacheFactory = Callable[[], Any]
GeometryCacheFactory = Callable[[], Any]
GeometryEngineFactory = Callable[[], Any]
Encoder3DFactory = Callable[[], Any]


def _default_feature_extractor_factory() -> Any:
    return FeatureExtractor()


def _default_feature_cache_factory() -> Any:
    from src.core.feature_cache import get_feature_cache

    return get_feature_cache()


def _default_geometry_cache_factory() -> Any:
    from src.core.geometry.cache import get_feature_cache

    return get_feature_cache()


def _default_geometry_engine_factory() -> Any:
    from src.core.geometry.engine import get_geometry_engine

    return get_geometry_engine()


def _default_encoder_3d_factory() -> Any:
    from src.ml.vision_3d import get_3d_encoder

    return get_3d_encoder()


async def run_feature_pipeline(
    *,
    extract_features: bool,
    file_format: str,
    file_name: str,
    content: bytes,
    doc: Any,
    started_at: float,
    stage_times: Mapping[str, float],
    feature_extractor_factory: Optional[FeatureExtractorFactory] = None,
    feature_cache_factory: Optional[FeatureCacheFactory] = None,
    geometry_cache_factory: Optional[GeometryCacheFactory] = None,
    geometry_engine_factory: Optional[GeometryEngineFactory] = None,
    encoder_3d_factory: Optional[Encoder3DFactory] = None,
    logger_instance: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    active_logger = logger_instance or logger
    results_patch: Dict[str, Any] = {}
    features: Dict[str, Any] = {"geometric": [], "semantic": []}
    features_3d: Dict[str, Any] = {}
    features_stage_duration: Optional[float] = None
    features_3d_stage_duration: Optional[float] = None

    if extract_features and file_format in ["step", "stp", "iges", "igs"]:
        try:
            geo_start = time.time()
            feature_cache_3d = (geometry_cache_factory or _default_geometry_cache_factory)()
            feature_version_3d = "l4_v1"
            cache_key_3d = feature_cache_3d.generate_key(content, feature_version_3d)
            cached_3d = feature_cache_3d.get(cache_key_3d)

            if cached_3d:
                features_3d = cached_3d
                active_logger.info("3D Feature Cache HIT for %s", file_name)
            else:
                geo_engine = (geometry_engine_factory or _default_geometry_engine_factory)()
                shape = geo_engine.load_step(content, file_name=file_name)
                if shape:
                    features_3d = geo_engine.extract_brep_features(shape)
                    features_3d.update(geo_engine.extract_dfm_features(shape))
                    encoder = (encoder_3d_factory or _default_encoder_3d_factory)()
                    embedding_3d = encoder.encode(features_3d)
                    features_3d["embedding_vector"] = embedding_3d
                    feature_cache_3d.set(cache_key_3d, features_3d)

            if "embedding_vector" in features_3d:
                results_patch["features_3d"] = {
                    key: value
                    for key, value in features_3d.items()
                    if key != "embedding_vector"
                }
                results_patch["features_3d"]["embedding_dim"] = len(
                    features_3d["embedding_vector"]
                )
            features_3d_stage_duration = time.time() - geo_start
        except Exception as exc:
            active_logger.error("L3 Analysis failed: %s", exc)

    if extract_features:
        from src.utils.analysis_metrics import (
            feature_cache_hits_total,
            feature_cache_lookup_seconds,
            feature_cache_miss_total,
            feature_cache_size,
        )

        feature_version = os.getenv("FEATURE_VERSION", "v1")
        content_hash_full = hashlib.sha256(content).hexdigest()
        cache_key = f"{content_hash_full}:{feature_version}:layout_v2"
        feature_cache = (feature_cache_factory or _default_feature_cache_factory)()
        lookup_start = time.time()
        cached_vector = feature_cache.get(cache_key)
        feature_cache_lookup_seconds.observe(time.time() - lookup_start)

        extractor = (feature_extractor_factory or _default_feature_extractor_factory)()
        combined_vec: Optional[List[float]] = None
        if cached_vector is not None:
            feature_cache_hits_total.inc()
            features = extractor.rehydrate(cached_vector, version=feature_version)
            combined_vec = cached_vector
        else:
            feature_cache_miss_total.inc()
            features = await extractor.extract(doc, brep_features=features_3d)
            try:
                combined_vec = extractor.flatten(features)
                feature_cache.set(cache_key, combined_vec)
                feature_cache_size.set(feature_cache.size())
            except Exception:
                pass

        if combined_vec is None:
            try:
                combined_vec = extractor.flatten(features)
            except Exception:
                combined_vec = []

        feature_slots = extractor.slots(feature_version)
        results_patch["features"] = {
            "geometric": [float(x) for x in features["geometric"]],
            "semantic": [float(x) for x in features["semantic"]],
            "combined": [float(x) for x in combined_vec],
            "dimension": len(features["geometric"]) + len(features["semantic"]),
            "feature_version": feature_version,
            "feature_slots": feature_slots,
            "cache_hit": cached_vector is not None,
        }
        features_stage_duration = time.time() - started_at - sum(stage_times.values())

    return {
        "features": features,
        "features_3d": features_3d,
        "results_patch": results_patch,
        "features_stage_duration": features_stage_duration,
        "features_3d_stage_duration": features_3d_stage_duration,
    }


__all__ = ["run_feature_pipeline"]
