#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional


_DATE_DIR_PATTERN = re.compile(r"^\d{8}$")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _parse_day(day_text: str) -> Optional[dt.date]:
    token = _safe_text(day_text)
    if not token or not _DATE_DIR_PATTERN.fullmatch(token):
        return None
    try:
        return dt.datetime.strptime(token, "%Y%m%d").date()
    except ValueError:
        return None


def _resolve_today(day_text: str) -> dt.date:
    parsed = _parse_day(day_text)
    if parsed is not None:
        return parsed
    return dt.datetime.now(tz=dt.timezone.utc).date()


def _discover_date_dirs(experiments_root: Path) -> List[Path]:
    if not experiments_root.exists() or not experiments_root.is_dir():
        return []
    candidates: List[Path] = []
    for child in sorted(experiments_root.iterdir(), key=lambda p: p.name):
        if child.is_dir() and _DATE_DIR_PATTERN.fullmatch(child.name):
            candidates.append(child)
    return candidates


def _allocate_archive_path(archive_root: Path, token: str, stamp: str) -> Path:
    target = archive_root / f"{token}_{stamp}.tar.gz"
    if not target.exists():
        return target
    idx = 2
    while True:
        candidate = archive_root / f"{token}_{stamp}_{idx}.tar.gz"
        if not candidate.exists():
            return candidate
        idx += 1


def archive_experiment_dirs(
    *,
    experiments_root: str,
    archive_root: str,
    explicit_dirs: List[str],
    keep_latest_days: int,
    dry_run: bool,
    delete_source: bool,
    require_exists: bool,
    today_text: str,
    manifest_json: str,
) -> Dict[str, Any]:
    experiments_dir = Path(_safe_text(experiments_root) or "reports/experiments")
    archive_dir = Path(_safe_text(archive_root) or "reports/archives/experiments")
    archive_dir.mkdir(parents=True, exist_ok=True)

    today = _resolve_today(today_text)
    keep_days = max(int(keep_latest_days), 0)
    cutoff = today - dt.timedelta(days=keep_days)
    stamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d_%H%M%S")

    discovered = _discover_date_dirs(experiments_dir)
    by_token = {item.name: item for item in discovered}
    selected: List[Path] = []
    rows: List[Dict[str, Any]] = []
    missing_count = 0
    archived_count = 0
    deleted_count = 0

    explicit_tokens = [_safe_text(token) for token in explicit_dirs if _safe_text(token)]
    if explicit_tokens:
        for token in explicit_tokens:
            src = by_token.get(token)
            if src is None:
                missing_count += 1
                rows.append(
                    {
                        "token": token,
                        "source_dir": str(experiments_dir / token),
                        "archive_file": "",
                        "status": "missing_source",
                        "archived": False,
                        "deleted": False,
                    }
                )
                continue
            selected.append(src)
    else:
        for src in discovered:
            parsed = _parse_day(src.name)
            if parsed is None:
                continue
            if parsed <= cutoff:
                selected.append(src)

    selected = sorted(selected, key=lambda p: p.name)
    for src in selected:
        token = src.name
        archive_file = _allocate_archive_path(archive_dir, token, stamp)
        if dry_run:
            rows.append(
                {
                    "token": token,
                    "source_dir": str(src),
                    "archive_file": str(archive_file),
                    "status": "dry_run",
                    "archived": False,
                    "deleted": False,
                }
            )
            continue

        with tarfile.open(archive_file, mode="w:gz") as handle:
            handle.add(src, arcname=token)
        archived_count += 1

        removed = False
        if delete_source:
            shutil.rmtree(src)
            removed = True
            deleted_count += 1

        rows.append(
            {
                "token": token,
                "source_dir": str(src),
                "archive_file": str(archive_file),
                "status": "archived",
                "archived": True,
                "deleted": removed,
            }
        )

    status = "ok"
    if require_exists and missing_count > 0:
        status = "failed"

    manifest: Dict[str, Any] = {
        "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "status": status,
        "experiments_root": str(experiments_dir),
        "archive_root": str(archive_dir),
        "today": today.strftime("%Y%m%d"),
        "keep_latest_days": keep_days,
        "cutoff_date": cutoff.strftime("%Y%m%d"),
        "explicit_dirs": explicit_tokens,
        "dry_run": bool(dry_run),
        "delete_source": bool(delete_source),
        "require_exists": bool(require_exists),
        "selected_count": len(selected),
        "archived_count": archived_count,
        "deleted_count": deleted_count,
        "missing_count": missing_count,
        "rows": rows,
    }

    manifest_target = Path(_safe_text(manifest_json)) if _safe_text(manifest_json) else (
        archive_dir / f"archive_manifest_{stamp}.json"
    )
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_json"] = str(manifest_target)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Archive reports/experiments date directories to .tar.gz, optionally "
            "deleting source directories after archive."
        )
    )
    parser.add_argument(
        "--experiments-root",
        default="reports/experiments",
        help="Root directory containing YYYYMMDD experiment subdirectories.",
    )
    parser.add_argument(
        "--archive-root",
        required=True,
        help="Destination root where tar.gz archives are written.",
    )
    parser.add_argument(
        "--dir",
        action="append",
        default=[],
        help="Specific YYYYMMDD directory to archive (repeatable).",
    )
    parser.add_argument(
        "--keep-latest-days",
        type=int,
        default=7,
        help=(
            "When --dir is not provided, archive discovered date dirs older than "
            "today - keep-latest-days."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan archive actions without writing tar files or deleting source dirs.",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete source date directories after successful archive.",
    )
    parser.add_argument(
        "--require-exists",
        action="store_true",
        help="Return non-zero if any explicit --dir path is missing.",
    )
    parser.add_argument(
        "--today",
        default="",
        help="Optional YYYYMMDD override used for deterministic selection.",
    )
    parser.add_argument(
        "--manifest-json",
        default="",
        help="Optional manifest output path.",
    )
    args = parser.parse_args()

    manifest = archive_experiment_dirs(
        experiments_root=str(args.experiments_root),
        archive_root=str(args.archive_root),
        explicit_dirs=[str(item) for item in list(args.dir or [])],
        keep_latest_days=int(args.keep_latest_days),
        dry_run=bool(args.dry_run),
        delete_source=bool(args.delete_source),
        require_exists=bool(args.require_exists),
        today_text=str(args.today),
        manifest_json=str(args.manifest_json),
    )
    print(f"status={manifest.get('status')}")
    print(f"selected_count={manifest.get('selected_count')}")
    print(f"archived_count={manifest.get('archived_count')}")
    print(f"deleted_count={manifest.get('deleted_count')}")
    print(f"missing_count={manifest.get('missing_count')}")
    print(f"manifest_json={manifest.get('manifest_json')}")
    if str(manifest.get("status")) != "ok":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
