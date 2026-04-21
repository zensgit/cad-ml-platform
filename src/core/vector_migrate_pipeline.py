from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Mapping, MutableMapping

from fastapi import HTTPException


async def run_vector_migrate_pipeline(
    *,
    payload: Any,
    vector_store: MutableMapping[str, Any],
    vector_meta: MutableMapping[str, Any],
    qdrant_store: Any,
    feature_extractor_cls: type[Any],
    prepare_vector_for_upgrade_fn: Callable[..., tuple[list[float], list[float], str]],
    vector_layout_base: str,
    vector_layout_l3: str,
    dimension_delta_metric: Any,
    migrate_total_metric: Any,
    analysis_error_code_total_metric: Any,
    error_code_cls: Any,
    build_error_fn: Callable[..., dict[str, Any]],
    item_cls: type[Any],
    response_cls: type[Any],
    history: list[dict[str, Any]],
    uuid4_fn: Callable[[], Any],
    utcnow_fn: Callable[[], datetime],
) -> Any:
    allowed_versions = {"v1", "v2", "v3", "v4"}
    if payload.to_version not in allowed_versions:
        err = build_error_fn(
            error_code_cls.INPUT_VALIDATION_FAILED,
            stage="vector_migrate",
            message="Unsupported target feature version",
            to_version=payload.to_version,
            allowed=list(sorted(allowed_versions)),
        )
        analysis_error_code_total_metric.labels(code=error_code_cls.INPUT_VALIDATION_FAILED.value).inc()
        raise HTTPException(status_code=422, detail=err)

    migration_id = str(uuid4_fn())
    started_at = utcnow_fn()
    target_version = payload.to_version
    items: list[Any] = []
    migrated = 0
    skipped = 0
    dry_run_total = 0
    extractor = feature_extractor_cls(feature_version=target_version)

    for vid in payload.ids:
        if qdrant_store is not None:
            target = await qdrant_store.get_vector(vid)
            if target is None:
                items.append(item_cls(id=vid, status="not_found", error="not_found"))
                skipped += 1
                migrate_total_metric.labels(status="not_found").inc()
                continue
            meta = dict(target.metadata or {})
            vec = list(target.vector or [])
        else:
            if vid not in vector_store:
                items.append(item_cls(id=vid, status="not_found", error="not_found"))
                skipped += 1
                migrate_total_metric.labels(status="not_found").inc()
                continue
            meta = vector_meta.get(vid, {})
            vec = vector_store[vid]

        from_version = meta.get("feature_version", "v1")
        if from_version == target_version:
            items.append(
                item_cls(
                    id=vid,
                    status="skipped",
                    from_version=from_version,
                    to_version=target_version,
                )
            )
            skipped += 1
            migrate_total_metric.labels(status="skipped").inc()
            continue

        dimension_before = len(vec)
        try:
            base_vector, l3_tail, _ = prepare_vector_for_upgrade_fn(
                extractor,
                vec,
                meta,
                from_version,
            )
            new_features = extractor.upgrade_vector(base_vector, current_version=from_version)
            if l3_tail:
                new_features = new_features + l3_tail
            dimension_after = len(new_features)
            dimension_delta = dimension_after - dimension_before
            dimension_delta_metric.observe(dimension_delta)

            if payload.dry_run:
                items.append(
                    item_cls(
                        id=vid,
                        status="dry_run",
                        from_version=from_version,
                        to_version=target_version,
                        dimension_before=dimension_before,
                        dimension_after=dimension_after,
                    )
                )
                dry_run_total += 1
                migrate_total_metric.labels(status="dry_run").inc()
                continue

            vector_store[vid] = new_features
            meta["feature_version"] = target_version
            expected_2d_dim = extractor.expected_dim(target_version)
            meta["geometric_dim"] = str(expected_2d_dim - 2)
            meta["semantic_dim"] = "2"
            meta["total_dim"] = str(len(new_features))
            meta["vector_layout"] = vector_layout_l3 if l3_tail else vector_layout_base
            if l3_tail:
                meta["l3_3d_dim"] = str(len(l3_tail))
            else:
                meta.pop("l3_3d_dim", None)
            if qdrant_store is not None:
                await qdrant_store.register_vector(vid, new_features, metadata=meta)
            else:
                vector_store[vid] = new_features
                vector_meta[vid] = meta

            if (from_version, target_version) in {
                ("v4", "v3"),
                ("v4", "v2"),
                ("v4", "v1"),
                ("v3", "v2"),
                ("v3", "v1"),
                ("v2", "v1"),
            }:
                items.append(
                    item_cls(
                        id=vid,
                        status="downgraded",
                        from_version=from_version,
                        to_version=target_version,
                        dimension_before=dimension_before,
                        dimension_after=dimension_after,
                    )
                )
                migrate_total_metric.labels(status="downgraded").inc()
            else:
                migrated += 1
                items.append(
                    item_cls(
                        id=vid,
                        status="migrated",
                        from_version=from_version,
                        to_version=target_version,
                        dimension_before=dimension_before,
                        dimension_after=dimension_after,
                    )
                )
                migrate_total_metric.labels(status="migrated").inc()
        except Exception as exc:
            items.append(item_cls(id=vid, status="error", error=str(exc)))
            skipped += 1
            migrate_total_metric.labels(status="error").inc()

    finished_at = utcnow_fn()
    history.append(
        {
            "migration_id": migration_id,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "total": len(payload.ids),
            "migrated": migrated,
            "skipped": skipped,
            "dry_run_total": dry_run_total,
            "counts": {
                "migrated": migrated,
                "skipped": skipped,
                "dry_run": dry_run_total,
                "downgraded": sum(1 for item in items if item.status == "downgraded"),
                "error": sum(1 for item in items if item.status == "error"),
                "not_found": sum(1 for item in items if item.status == "not_found"),
            },
        }
    )
    if len(history) > 10:
        history.pop(0)

    return response_cls(
        total=len(payload.ids),
        migrated=migrated,
        skipped=skipped,
        items=items,
        migration_id=migration_id,
        started_at=started_at,
        finished_at=finished_at,
        dry_run_total=dry_run_total if payload.dry_run else None,
    )


__all__ = ["run_vector_migrate_pipeline"]
