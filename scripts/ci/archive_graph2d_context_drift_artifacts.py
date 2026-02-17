#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


def _resolve_date(date_text: str) -> str:
    token = str(date_text).strip()
    if token:
        return token
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d")


def _safe_text(value: Any) -> str:
    return str(value).strip()


def archive_artifacts(
    *,
    artifacts: List[str],
    output_root: str,
    date_text: str,
    bucket: str,
    require_exists: bool,
    manifest_path: str,
) -> Dict[str, Any]:
    date_token = _resolve_date(date_text)
    bucket_token = _safe_text(bucket) or "graph2d_context_drift"
    archive_dir = Path(_safe_text(output_root) or "reports/experiments") / date_token / bucket_token
    archive_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    missing_count = 0
    copied_count = 0
    for raw in artifacts:
        src_text = _safe_text(raw)
        if not src_text:
            continue
        src = Path(src_text)
        exists = src.exists() and src.is_file()
        if not exists:
            missing_count += 1
            rows.append(
                {
                    "source": src_text,
                    "exists": False,
                    "copied": False,
                    "destination": "",
                }
            )
            continue

        dst = archive_dir / src.name
        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            idx = 2
            while True:
                candidate = archive_dir / f"{stem}_{idx}{suffix}"
                if not candidate.exists():
                    dst = candidate
                    break
                idx += 1
        shutil.copy2(src, dst)
        copied_count += 1
        rows.append(
            {
                "source": src_text,
                "exists": True,
                "copied": True,
                "destination": str(dst),
            }
        )

    manifest = {
        "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "archive_dir": str(archive_dir),
        "date": date_token,
        "bucket": bucket_token,
        "require_exists": bool(require_exists),
        "copied_count": copied_count,
        "missing_count": missing_count,
        "rows": rows,
    }
    if require_exists and missing_count > 0:
        manifest["status"] = "failed"
    else:
        manifest["status"] = "ok"

    manifest_target = Path(_safe_text(manifest_path)) if _safe_text(manifest_path) else (archive_dir / "archive_manifest.json")
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_json"] = str(manifest_target)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Archive Graph2D context drift artifacts into reports/experiments/<date>/."
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Artifact file path to archive (repeatable).",
    )
    parser.add_argument(
        "--output-root",
        default="reports/experiments",
        help="Archive root directory.",
    )
    parser.add_argument(
        "--date",
        default="",
        help="Date token (YYYYMMDD); default is UTC today.",
    )
    parser.add_argument(
        "--bucket",
        default="graph2d_context_drift",
        help="Subdirectory name under date folder.",
    )
    parser.add_argument(
        "--manifest-json",
        default="",
        help="Optional manifest output path.",
    )
    parser.add_argument(
        "--require-exists",
        action="store_true",
        help="Fail when any artifact is missing.",
    )
    args = parser.parse_args()

    manifest = archive_artifacts(
        artifacts=[str(item) for item in list(args.artifact or [])],
        output_root=str(args.output_root),
        date_text=str(args.date),
        bucket=str(args.bucket),
        require_exists=bool(args.require_exists),
        manifest_path=str(args.manifest_json),
    )
    print(f"archive_dir={manifest.get('archive_dir')}")
    print(f"copied_count={manifest.get('copied_count')}")
    print(f"missing_count={manifest.get('missing_count')}")
    print(f"manifest_json={manifest.get('manifest_json')}")
    if str(manifest.get("status")) != "ok":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
