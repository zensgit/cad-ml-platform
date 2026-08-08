#!/usr/bin/env python3
"""Backup / cleanup / list helpers for ReviewReuse filesystem task store.

Layout (see FilesystemReviewReuseStore)::

  {store_dir}/{tenant}/tasks/*.json
  {store_dir}/{tenant}/idempotency.json

Does not enable decisions, touch training JSONL, or call network services.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _tenant_dirs(store_dir: Path) -> List[Path]:
    if not store_dir.is_dir():
        return []
    return sorted(p for p in store_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def _task_files(tenant_dir: Path) -> List[Path]:
    tasks = tenant_dir / "tasks"
    if not tasks.is_dir():
        return []
    return sorted(tasks.glob("*.json"))


def _newest_mtime(tenant_dir: Path) -> Optional[float]:
    newest: Optional[float] = None
    paths: List[Path] = list(_task_files(tenant_dir))
    idem = tenant_dir / "idempotency.json"
    if idem.is_file():
        paths.append(idem)
    if not paths:
        try:
            return tenant_dir.stat().st_mtime
        except OSError:
            return None
    for p in paths:
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if newest is None or m > newest:
            newest = m
    return newest


def _newest_task_mtime(tenant_dir: Path) -> Optional[float]:
    """Newest mtime among task JSON files only (excludes idempotency)."""
    newest: Optional[float] = None
    for p in _task_files(tenant_dir):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if newest is None or m > newest:
            newest = m
    return newest


def collect_tenant_summaries(
    store_dir: Path, *, now: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Return per-tenant task_count and age_days of newest task."""
    now_ts = time.time() if now is None else now
    store_dir = store_dir.resolve()
    rows: List[Dict[str, Any]] = []
    for tdir in _tenant_dirs(store_dir):
        task_files = _task_files(tdir)
        newest = _newest_task_mtime(tdir)
        age_days: Optional[float]
        if newest is None:
            age_days = None
        else:
            age_days = (now_ts - newest) / 86400.0
        rows.append(
            {
                "tenant": tdir.name,
                "task_count": len(task_files),
                "age_days": age_days,
            }
        )
    return rows


def cmd_list(store_dir: Path, *, as_json: bool = False) -> int:
    store_dir = store_dir.resolve()
    rows = collect_tenant_summaries(store_dir)
    if as_json:
        payload = {
            "store_dir": str(store_dir),
            "tenants": [
                {
                    "tenant": r["tenant"],
                    "task_count": r["task_count"],
                    "age_days": (
                        None
                        if r["age_days"] is None
                        else round(float(r["age_days"]), 4)
                    ),
                }
                for r in rows
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if not store_dir.is_dir():
        print(f"store_dir missing (empty): {store_dir}", file=sys.stderr)
    for r in rows:
        age = r["age_days"]
        age_s = "n/a" if age is None else f"{float(age):.1f}"
        print(f"tenant={r['tenant']} tasks={r['task_count']} age_days={age_s}")
    print(f"tenants={len(rows)} store_dir={store_dir}")
    return 0


def cmd_backup(store_dir: Path, out_dir: Path) -> int:
    store_dir = store_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not store_dir.is_dir():
        print(f"store_dir missing (nothing to backup): {store_dir}", file=sys.stderr)
        return 1
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = out_dir / f"review_reuse_store_{ts}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(store_dir, arcname=store_dir.name)
    print(f"backup={archive}")
    print(f"source={store_dir}")
    return 0


def cmd_cleanup(
    store_dir: Path,
    *,
    older_than_days: float,
    dry_run: bool,
    tenant: Optional[str],
) -> int:
    store_dir = store_dir.resolve()
    if not store_dir.is_dir():
        print(f"store_dir missing: {store_dir}", file=sys.stderr)
        return 1
    cutoff = time.time() - (older_than_days * 86400.0)
    tenants = _tenant_dirs(store_dir)
    if tenant:
        tenants = [t for t in tenants if t.name == tenant]
        if not tenants:
            print(f"tenant not found: {tenant}", file=sys.stderr)
            return 1

    removed = 0
    listed = 0
    for tdir in tenants:
        newest = _newest_mtime(tdir)
        if newest is None:
            continue
        if newest > cutoff:
            continue
        listed += 1
        age_days = (time.time() - newest) / 86400.0
        if dry_run:
            print(f"would_delete tenant={tdir.name} age_days={age_days:.1f} path={tdir}")
        else:
            shutil.rmtree(tdir)
            print(f"deleted tenant={tdir.name} age_days={age_days:.1f}")
            removed += 1

    mode = "dry_run" if dry_run else "apply"
    print(f"cleanup mode={mode} candidates={listed} deleted={removed} older_than_days={older_than_days}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    default_store = os.getenv("REVIEW_REUSE_STORE_DIR", "data/review_reuse_tasks")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_backup = sub.add_parser("backup", help="Create tar.gz of store_dir")
    p_backup.add_argument("--store-dir", type=Path, default=Path(default_store))
    p_backup.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/review_reuse_backups"),
        help="Directory for timestamped archives",
    )

    p_clean = sub.add_parser("cleanup", help="Delete tenant dirs older than N days")
    p_clean.add_argument("--store-dir", type=Path, default=Path(default_store))
    p_clean.add_argument("--older-than-days", type=float, default=30.0)
    p_clean.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="List only (default true)",
    )
    p_clean.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete (overrides --dry-run)",
    )
    p_clean.add_argument("--tenant", default=None, help="Limit to one tenant segment")

    p_list = sub.add_parser(
        "list",
        help="List tenants with task count and age_days of newest task",
    )
    p_list.add_argument("--store-dir", type=Path, default=Path(default_store))
    p_list.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Machine-readable JSON output",
    )

    args = parser.parse_args(argv)
    if args.cmd == "backup":
        return cmd_backup(args.store_dir, args.out_dir)
    if args.cmd == "cleanup":
        dry = not args.apply
        return cmd_cleanup(
            args.store_dir,
            older_than_days=args.older_than_days,
            dry_run=dry,
            tenant=args.tenant,
        )
    if args.cmd == "list":
        return cmd_list(args.store_dir, as_json=args.as_json)
    return 2


if __name__ == "__main__":
    sys.exit(main())
