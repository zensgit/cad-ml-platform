from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_STORE_DIR_ENV = "ANALYSIS_RESULT_STORE_DIR"
_STORE_TTL_ENV = "ANALYSIS_RESULT_STORE_TTL_SECONDS"
_STORE_MAX_FILES_ENV = "ANALYSIS_RESULT_STORE_MAX_FILES"
_VALID_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _get_store_dir(*, create: bool) -> Optional[Path]:
    raw = os.getenv(_STORE_DIR_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.exists() and not path.is_dir():
        logger.warning(
            "analysis_result_store_invalid_dir",
            extra={"error": f"Path is not a directory: {path}"},
        )
        return None
    if not path.exists():
        if not create:
            return None
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - filesystem edge
            logger.warning(
                "analysis_result_store_init_failed",
                extra={"error": str(exc)},
            )
            return None
    return path


def _resolve_path(analysis_id: str, *, create: bool) -> Optional[Path]:
    if not _VALID_ID_RE.fullmatch(analysis_id or ""):
        return None
    store_dir = _get_store_dir(create=create)
    if store_dir is None:
        return None
    return store_dir / f"{analysis_id}.json"


def _get_env_int(name: str) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _format_mtime(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _record_cleanup_metrics(status: str, *, deleted: int, total_files: int) -> None:
    try:
        from src.utils.analysis_metrics import (
            analysis_result_cleanup_deleted_total,
            analysis_result_cleanup_total,
            analysis_result_store_files,
        )

        analysis_result_cleanup_total.labels(status=status).inc()
        if deleted:
            analysis_result_cleanup_deleted_total.inc(deleted)
        analysis_result_store_files.set(max(total_files - deleted, 0))
    except Exception:
        pass


def _list_result_files(store_dir: Path) -> list[tuple[str, Path, float]]:
    entries: list[tuple[str, Path, float]] = []
    for path in store_dir.glob("*.json"):
        if not path.is_file():
            continue
        analysis_id = path.stem
        if not _VALID_ID_RE.fullmatch(analysis_id):
            continue
        try:
            mtime = path.stat().st_mtime
        except Exception:
            continue
        entries.append((analysis_id, path, mtime))
    entries.sort(key=lambda item: item[0])
    return entries


def _select_candidates(
    files: list[tuple[str, Path, float]],
    max_age_seconds: Optional[int],
    max_files: Optional[int],
) -> tuple[list[tuple[str, Path, float]], int, int]:
    expired: list[tuple[str, Path, float]] = []
    if max_age_seconds:
        now = time.time()
        expired = [item for item in files if now - item[2] > max_age_seconds]

    expired_paths = {item[1] for item in expired}
    remaining = [item for item in files if item[1] not in expired_paths]

    overflow: list[tuple[str, Path, float]] = []
    if max_files and len(remaining) > max_files:
        overflow_count = len(remaining) - max_files
        overflow = sorted(remaining, key=lambda item: item[2])[:overflow_count]

    return expired + overflow, len(expired), len(overflow)


async def store_analysis_result(analysis_id: str, result: dict) -> bool:
    path = _resolve_path(analysis_id, create=True)
    if path is None:
        return False
    tmp_path = path.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=True)
        os.replace(tmp_path, path)
        return True
    except Exception as exc:
        logger.warning(
            "analysis_result_store_write_failed",
            extra={"error": str(exc)},
        )
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return False


async def load_analysis_result(analysis_id: str) -> Optional[dict[str, Any]]:
    path = _resolve_path(analysis_id, create=False)
    if path is None or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logger.warning(
            "analysis_result_store_read_failed",
            extra={"error": str(exc)},
        )
        return None


async def cleanup_analysis_results(
    *,
    max_age_seconds: Optional[int] = None,
    max_files: Optional[int] = None,
    dry_run: bool = False,
    sample_limit: int = 0,
) -> dict[str, Any]:
    store_dir = _get_store_dir(create=False)
    if store_dir is None:
        result = {
            "status": "disabled",
            "total_files": 0,
            "eligible_count": 0,
            "expired_count": 0,
            "overflow_count": 0,
            "deleted_count": 0,
            "max_age_seconds": max_age_seconds,
            "max_files": max_files,
            "sample_ids": [],
            "message": "ANALYSIS_RESULT_STORE_DIR not configured",
        }
        _record_cleanup_metrics(result["status"], deleted=0, total_files=0)
        return result

    effective_age = max_age_seconds if max_age_seconds is not None else _get_env_int(_STORE_TTL_ENV)
    effective_max_files = (
        max_files if max_files is not None else _get_env_int(_STORE_MAX_FILES_ENV)
    )

    files = _list_result_files(store_dir)
    total_files = len(files)

    if effective_age is None and effective_max_files is None:
        result = {
            "status": "skipped",
            "total_files": total_files,
            "eligible_count": 0,
            "expired_count": 0,
            "overflow_count": 0,
            "deleted_count": 0,
            "max_age_seconds": effective_age,
            "max_files": effective_max_files,
            "sample_ids": [],
            "message": "No cleanup policy configured",
        }
        _record_cleanup_metrics(result["status"], deleted=0, total_files=total_files)
        return result

    candidates, expired_count, overflow_count = _select_candidates(
        files, effective_age, effective_max_files
    )
    eligible_count = len(candidates)
    sample_ids = [item[0] for item in candidates[:sample_limit]] if sample_limit else []

    if eligible_count == 0:
        result = {
            "status": "dry_run" if dry_run else "ok",
            "total_files": total_files,
            "eligible_count": 0,
            "expired_count": expired_count,
            "overflow_count": overflow_count,
            "deleted_count": 0,
            "max_age_seconds": effective_age,
            "max_files": effective_max_files,
            "sample_ids": sample_ids,
            "message": "No analysis results eligible for cleanup",
        }
        _record_cleanup_metrics(result["status"], deleted=0, total_files=total_files)
        return result

    if dry_run:
        result = {
            "status": "dry_run",
            "total_files": total_files,
            "eligible_count": eligible_count,
            "expired_count": expired_count,
            "overflow_count": overflow_count,
            "deleted_count": 0,
            "max_age_seconds": effective_age,
            "max_files": effective_max_files,
            "sample_ids": sample_ids,
            "message": f"Would delete {eligible_count} analysis results",
        }
        _record_cleanup_metrics(result["status"], deleted=0, total_files=total_files)
        return result

    deleted_count = 0
    for _, path, _ in candidates:
        try:
            path.unlink()
            deleted_count += 1
        except Exception as exc:
            logger.warning(
                "analysis_result_store_delete_failed",
                extra={"error": str(exc)},
            )

    result = {
        "status": "ok",
        "total_files": total_files,
        "eligible_count": eligible_count,
        "expired_count": expired_count,
        "overflow_count": overflow_count,
        "deleted_count": deleted_count,
        "max_age_seconds": effective_age,
        "max_files": effective_max_files,
        "sample_ids": sample_ids,
        "message": f"Deleted {deleted_count} analysis results",
    }
    _record_cleanup_metrics(result["status"], deleted=deleted_count, total_files=total_files)
    return result


def get_analysis_result_store_stats() -> dict[str, Any]:
    store_dir = _get_store_dir(create=False)
    effective_age = _get_env_int(_STORE_TTL_ENV)
    effective_max_files = _get_env_int(_STORE_MAX_FILES_ENV)

    if store_dir is None:
        return {
            "enabled": False,
            "path": None,
            "total_files": 0,
            "oldest_mtime": None,
            "newest_mtime": None,
            "max_age_seconds": effective_age,
            "max_files": effective_max_files,
        }

    files = _list_result_files(store_dir)
    total_files = len(files)
    mtimes = [item[2] for item in files]
    oldest_ts = min(mtimes) if mtimes else None
    newest_ts = max(mtimes) if mtimes else None
    try:
        from src.utils.analysis_metrics import analysis_result_store_files

        analysis_result_store_files.set(total_files)
    except Exception:
        pass

    return {
        "enabled": True,
        "path": str(store_dir),
        "total_files": total_files,
        "oldest_mtime": _format_mtime(oldest_ts),
        "newest_mtime": _format_mtime(newest_ts),
        "max_age_seconds": effective_age,
        "max_files": effective_max_files,
    }


__all__ = [
    "cleanup_analysis_results",
    "get_analysis_result_store_stats",
    "load_analysis_result",
    "store_analysis_result",
]
