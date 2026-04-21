from __future__ import annotations

from typing import Any, Awaitable, Callable

from fastapi import HTTPException


async def run_vector_migration_preview_pipeline(
    *,
    to_version: str,
    limit: int,
    response_cls: type[Any],
    error_code_cls: Any,
    build_error_fn: Callable[..., dict[str, Any]],
    get_qdrant_store_fn: Callable[[], Any],
    collect_qdrant_preview_samples_fn: Callable[..., Awaitable[tuple[list[tuple[str, list[float], dict[str, Any]]], int, dict[str, int]]]],
    prepare_vector_for_upgrade_fn: Callable[..., tuple[list[float], list[float], str | None]],
    feature_extractor_cls: type[Any],
) -> Any:
    from src.utils.analysis_metrics import analysis_error_code_total

    allowed_versions = {"v1", "v2", "v3", "v4"}
    if to_version not in allowed_versions:
        err = build_error_fn(
            error_code_cls.INPUT_VALIDATION_FAILED,
            stage="migration_preview",
            message="Unsupported target feature version",
            to_version=to_version,
            allowed=list(allowed_versions),
        )
        analysis_error_code_total.labels(code=error_code_cls.INPUT_VALIDATION_FAILED.value).inc()
        raise HTTPException(status_code=422, detail=err)

    capped_limit = min(limit, 100)
    extractor = feature_extractor_cls(feature_version=to_version)

    qdrant_store = get_qdrant_store_fn()
    if qdrant_store is not None:
        preview_source, total_vectors, by_version = await collect_qdrant_preview_samples_fn(
            qdrant_store,
            limit=capped_limit,
        )
    else:
        from src.core.similarity import _VECTOR_META, _VECTOR_STORE

        by_version: dict[str, int] = {}
        total_vectors = len(_VECTOR_STORE)
        preview_source: list[tuple[str, list[float], dict[str, Any]]] = []

        for vector_id in _VECTOR_STORE.keys():
            meta = _VECTOR_META.get(vector_id, {})
            current_version = meta.get("feature_version", "v1")
            by_version[current_version] = by_version.get(current_version, 0) + 1

        for vector_id in list(_VECTOR_STORE.keys())[:capped_limit]:
            preview_source.append((vector_id, _VECTOR_STORE[vector_id], _VECTOR_META.get(vector_id, {})))

    preview_items: list[dict[str, Any]] = []
    dimension_changes = {"positive": 0, "negative": 0, "zero": 0}
    deltas: list[int] = []
    warnings: list[str] = []
    sampled = 0
    downgrade_pairs = {
        ("v4", "v3"),
        ("v4", "v2"),
        ("v4", "v1"),
        ("v3", "v2"),
        ("v3", "v1"),
        ("v2", "v1"),
    }

    for vector_id, vector, meta in preview_source:
        from_version = meta.get("feature_version", "v1")
        dimension_before = len(vector)

        if from_version == to_version:
            preview_items.append(
                {
                    "id": vector_id,
                    "status": "skipped",
                    "from_version": from_version,
                    "to_version": to_version,
                    "dimension_before": dimension_before,
                    "dimension_after": dimension_before,
                }
            )
            dimension_changes["zero"] += 1
            sampled += 1
            continue

        try:
            base_vector, l3_tail, _ = prepare_vector_for_upgrade_fn(
                extractor,
                vector,
                meta,
                from_version,
            )
            new_features = extractor.upgrade_vector(base_vector, current_version=from_version)
            if l3_tail:
                new_features = new_features + l3_tail
            dimension_after = len(new_features)
            dimension_delta = dimension_after - dimension_before
            deltas.append(dimension_delta)

            if dimension_delta > 0:
                dimension_changes["positive"] += 1
            elif dimension_delta < 0:
                dimension_changes["negative"] += 1
            else:
                dimension_changes["zero"] += 1

            preview_items.append(
                {
                    "id": vector_id,
                    "status": (
                        "downgrade_preview" if (from_version, to_version) in downgrade_pairs else "upgrade_preview"
                    ),
                    "from_version": from_version,
                    "to_version": to_version,
                    "dimension_before": dimension_before,
                    "dimension_after": dimension_after,
                }
            )
            sampled += 1
        except Exception as exc:
            preview_items.append({"id": vector_id, "status": "error_preview", "error": str(exc)})
            warnings.append(f"Vector {vector_id} migration would fail: {str(exc)}")
            sampled += 1

    migration_feasible = True
    total_sampled = max(sampled, 1)
    if dimension_changes["negative"] > total_sampled * 0.5:
        migration_feasible = False
        warnings.append("More than 50% of sampled vectors would lose dimensions")

    avg_delta = None
    median_delta = None
    if deltas:
        avg_delta = float(sum(deltas) / len(deltas))
        try:
            import statistics

            median_delta = float(statistics.median(deltas))
        except Exception:
            median_delta = float(deltas[len(deltas) // 2])
    if median_delta is not None and median_delta < -5:
        warnings.append("large_negative_skew")
    if avg_delta is not None and abs(avg_delta) > 10:
        warnings.append("growth_spike")
    if len(warnings) > capped_limit * 0.3:
        warnings.append(f"High error rate in preview: {len(warnings)}/{capped_limit}")

    return response_cls(
        total_vectors=total_vectors,
        by_version=by_version,
        preview_items=preview_items,
        estimated_dimension_changes=dimension_changes,
        migration_feasible=migration_feasible,
        warnings=warnings,
        avg_delta=avg_delta,
        median_delta=median_delta,
    )


__all__ = ["run_vector_migration_preview_pipeline"]
