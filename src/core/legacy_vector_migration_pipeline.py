from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from src.core.feature_extractor import FeatureExtractor
from src.models.cad_document import CadDocument
from src.utils.cache import get_cached_result


async def run_legacy_vector_migrate_pipeline(*, payload) -> dict[str, Any]:
    from src.core.similarity import _VECTOR_META, _VECTOR_STORE  # type: ignore

    items: list[dict[str, Any]] = []
    migrated = 0
    skipped = 0
    default_version = __import__("os").getenv("FEATURE_VERSION", "v1")
    started_at = datetime.now(timezone.utc)
    batch_id = str(uuid.uuid4())
    dry_run_total = 0

    for vid in payload.ids:
        meta = _VECTOR_META.get(vid, {})
        if vid not in _VECTOR_STORE:
            items.append(
                {
                    "id": vid,
                    "status": "not_found",
                    "to_version": payload.to_version,
                    "error": "vector_missing",
                }
            )
            skipped += 1
            continue

        from_version = meta.get("feature_version", default_version)
        original_dim = len(_VECTOR_STORE[vid])
        if from_version == payload.to_version:
            items.append(
                {
                    "id": vid,
                    "status": "skipped",
                    "from_version": from_version,
                    "to_version": payload.to_version,
                    "dimension_before": original_dim,
                    "dimension_after": original_dim,
                }
            )
            skipped += 1
            continue

        cached = await get_cached_result(f"analysis_result:{vid}")
        if not cached:
            items.append(
                {
                    "id": vid,
                    "status": "skipped",
                    "from_version": from_version,
                    "to_version": payload.to_version,
                    "error": "cached_result_missing",
                }
            )
            skipped += 1
            continue

        stats = cached.get("statistics", {})
        bbox = stats.get("bounding_box", {})
        doc = CadDocument(
            file_name=cached.get("file_name", vid),
            format=cached.get("file_format", "unknown"),
        )
        doc.bounding_box.min_x = bbox.get("min_x", 0.0)
        doc.bounding_box.min_y = bbox.get("min_y", 0.0)
        doc.bounding_box.min_z = bbox.get("min_z", 0.0)
        doc.bounding_box.max_x = bbox.get("max_x", 0.0)
        doc.bounding_box.max_y = bbox.get("max_y", 0.0)
        doc.bounding_box.max_z = bbox.get("max_z", 0.0)
        extractor = FeatureExtractor(feature_version=payload.to_version)

        try:
            new_features = await extractor.extract(doc)
            new_vector = extractor.flatten(new_features)
            if payload.dry_run:
                items.append(
                    {
                        "id": vid,
                        "status": "dry_run",
                        "from_version": from_version,
                        "to_version": payload.to_version,
                        "dimension_before": original_dim,
                        "dimension_after": len(new_vector),
                    }
                )
                skipped += 1
                dry_run_total += 1
            else:
                from src.core.vector_layouts import VECTOR_LAYOUT_BASE

                _VECTOR_STORE[vid] = [float(x) for x in new_vector]
                meta.update(
                    {
                        "feature_version": payload.to_version,
                        "geometric_dim": str(len(new_features.get("geometric", []))),
                        "semantic_dim": str(len(new_features.get("semantic", []))),
                        "total_dim": str(len(new_vector)),
                        "vector_layout": VECTOR_LAYOUT_BASE,
                    }
                )
                meta.pop("l3_3d_dim", None)
                items.append(
                    {
                        "id": vid,
                        "status": "migrated",
                        "from_version": from_version,
                        "to_version": payload.to_version,
                        "dimension_before": original_dim,
                        "dimension_after": len(new_vector),
                    }
                )
                migrated += 1
        except Exception as exc:
            items.append(
                {
                    "id": vid,
                    "status": "error",
                    "from_version": from_version,
                    "to_version": payload.to_version,
                    "error": str(exc),
                }
            )
            skipped += 1

    finished_at = datetime.now(timezone.utc)
    _record_legacy_migration_status(
        batch_id=batch_id,
        started_at=started_at,
        finished_at=finished_at,
        total=len(payload.ids),
        migrated=migrated,
        skipped=skipped,
        dry_run_total=dry_run_total,
    )
    return {
        "total": len(payload.ids),
        "migrated": migrated,
        "skipped": skipped,
        "items": items,
        "migration_id": batch_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "dry_run_total": dry_run_total,
    }


def run_legacy_vector_migration_status_pipeline() -> dict[str, Any]:
    from src.core.similarity import _VECTOR_META, _VECTOR_STORE  # type: ignore

    versions: dict[str, int] = {}
    for meta in _VECTOR_META.values():
        ver = meta.get("feature_version", "unknown")
        versions[ver] = versions.get(ver, 0) + 1

    status = _get_legacy_migration_status()
    return {
        "last_migration_id": status.get("last_migration_id"),
        "last_started_at": _parse_dt(status.get("last_started_at")),
        "last_finished_at": _parse_dt(status.get("last_finished_at")),
        "last_total": status.get("last_total"),
        "last_migrated": status.get("last_migrated"),
        "last_skipped": status.get("last_skipped"),
        "pending_vectors": len(_VECTOR_STORE),
        "feature_versions": versions,
        "history": status.get("history"),
    }


def _record_legacy_migration_status(
    *,
    batch_id: str,
    started_at: datetime,
    finished_at: datetime,
    total: int,
    migrated: int,
    skipped: int,
    dry_run_total: int,
) -> None:
    try:
        import src.core.similarity as _sim  # type: ignore

        if not hasattr(_sim, "_MIGRATION_STATUS"):
            _sim._MIGRATION_STATUS = {}
        hist = _sim._MIGRATION_STATUS.get("history", [])
        entry = {
            "migration_id": batch_id,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "total": total,
            "migrated": migrated,
            "skipped": skipped,
            "dry_run_total": dry_run_total,
        }
        hist.append(entry)
        if len(hist) > 10:
            hist = hist[-10:]
        _sim._MIGRATION_STATUS.update(
            {
                "last_migration_id": batch_id,
                "last_started_at": started_at.isoformat(),
                "last_finished_at": finished_at.isoformat(),
                "last_total": total,
                "last_migrated": migrated,
                "last_skipped": skipped,
                "last_dry_run_total": dry_run_total,
                "history": hist,
            }
        )
    except Exception:
        pass


def _get_legacy_migration_status() -> dict[str, Any]:
    try:
        import src.core.similarity as _sim  # type: ignore

        return getattr(_sim, "_MIGRATION_STATUS", {})
    except Exception:
        return {}


def _parse_dt(val: Optional[str]):
    if not val:
        return None
    try:
        return datetime.fromisoformat(val)
    except Exception:
        return None


__all__ = [
    "run_legacy_vector_migrate_pipeline",
    "run_legacy_vector_migration_status_pipeline",
]
