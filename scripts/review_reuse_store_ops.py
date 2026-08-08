#!/usr/bin/env python3
"""Backup / cleanup helpers for ReviewReuse filesystem task store.

Layout (see FilesystemReviewReuseStore)::

  {store_dir}/{tenant}/tasks/*.json
  {store_dir}/{tenant}/idempotency.json

Does not enable decisions, touch training JSONL, or call network services.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def _tenant_dirs(store_dir: Path) -> List[Path]:
    if not store_dir.is_dir():
        return []
    return sorted(p for p in store_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def _newest_mtime(tenant_dir: Path) -> Optional[float]:
    newest: Optional[float] = None
    tasks = tenant_dir / "tasks"
    paths: List[Path] = []
    if tasks.is_dir():
        paths.extend(tasks.glob("*.json"))
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
    return 2


if __name__ == "__main__":
    sys.exit(main())
