#!/usr/bin/env python3
"""Backup / cleanup / list helpers for ReviewReuse filesystem task store.

Layout (see FilesystemReviewReuseStore)::

  {store_dir}/{tenant}/tasks/*.json
  {store_dir}/{tenant}/idempotency.json

Does not enable decisions, touch training JSONL, or call network services.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_UNREADABLE = "__unreadable__"


def _tenant_dirs(store_dir: Path) -> List[Path]:
    if not store_dir.is_dir():
        return []
    return sorted(p for p in store_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def _task_attribution(tdir: Path) -> tuple[List[str], bool]:
    """Unique tenant_ids from readable tasks, plus unattributable-file flag."""
    found: List[str] = []
    seen = set()
    unattributable = False
    for path in _task_files(tdir):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            unattributable = True
            continue
        if not isinstance(data, dict):
            unattributable = True
            continue
        tid = data.get("tenant_id")
        if isinstance(tid, str) and tid:
            if tid not in seen:
                seen.add(tid)
                found.append(tid)
        else:
            unattributable = True
    return found, unattributable


def _tenant_ids_from_tasks(tdir: Path) -> List[str]:
    ids, _unattributable = _task_attribution(tdir)
    return ids


def _tenant_id_from_meta(tdir: Path) -> Optional[str]:
    """Sidecar tenant_id, ``_UNREADABLE`` if present-but-invalid, else None."""
    meta_path = tdir / "tenant_meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return _UNREADABLE
    if isinstance(meta, dict):
        tid = meta.get("tenant_id")
        if isinstance(tid, str) and tid:
            return tid
    return _UNREADABLE


def _is_mixed_tenant_dir(tdir: Path) -> bool:
    ids, unattributable = _task_attribution(tdir)
    if unattributable:
        return True
    if len(ids) > 1:
        return True
    meta = _tenant_id_from_meta(tdir)
    if meta == _UNREADABLE:
        return True
    return meta is not None and any(tid != meta for tid in ids)


def _recorded_tenant_id(tdir: Path) -> Optional[str]:
    """Original tenant_id from sidecar or a unique task payload, if any."""
    meta = _tenant_id_from_meta(tdir)
    if meta is not None and meta != _UNREADABLE:
        return meta
    ids = _tenant_ids_from_tasks(tdir)
    if len(ids) == 1:
        return ids[0]
    return None


def _tenant_label(tdir: Path) -> str:
    recorded = _recorded_tenant_id(tdir)
    return recorded if recorded is not None else tdir.name


def _dir_tenant_identities(tdir: Path) -> List[str]:
    """Every recorded tenant identity on a dir (meta + task payloads).

    Cleanup groups by this set so a mixed sibling labeled B that still
    holds A tasks blocks A's hashed directory before any rmtree.
    """
    ids: List[str] = []
    seen = set()
    meta = _tenant_id_from_meta(tdir)
    if meta is not None and meta != _UNREADABLE and meta not in seen:
        seen.add(meta)
        ids.append(meta)
    for tid in _tenant_ids_from_tasks(tdir):
        if tid not in seen:
            seen.add(tid)
            ids.append(tid)
    if ids:
        return ids
    return [_tenant_label(tdir)]


def _tenant_dir_key(tenant_id: str) -> str:
    return hashlib.sha256((tenant_id or "").encode("utf-8")).hexdigest()[:24]


def _is_hash_dirname(name: str) -> bool:
    return len(name) == 24 and all(c in "0123456789abcdef" for c in name)


def _unattributable_hash_dir(tdir: Path) -> bool:
    """Hashed dir whose meta/tasks do not name a tenant."""
    if not _is_hash_dirname(tdir.name):
        return False
    if _recorded_tenant_id(tdir) is not None:
        return False
    meta = _tenant_id_from_meta(tdir)
    _, unattributable = _task_attribution(tdir)
    return meta == _UNREADABLE or unattributable


def _attach_unreadable_hash_siblings(
    groups: Dict[str, List[Path]], tenants: List[Path]
) -> None:
    """Keep a corrupt hash sibling in the logical tenant group.

    Without this, ``--apply`` deletes the readable legacy directory and
    leaves the unreadable hashed directory behind.
    """
    idents = [ident for ident in groups if ident != _UNREADABLE]
    for tdir in tenants:
        if not _unattributable_hash_dir(tdir):
            continue
        for ident in idents:
            if tdir.name != _tenant_dir_key(ident):
                continue
            bucket = groups.setdefault(ident, [])
            if tdir not in bucket:
                bucket.append(tdir)


def _tenant_matches(tdir: Path, tenant: str) -> bool:
    """Match by original tenant identity, not a colliding hashed basename.

    A hashed dir named ``sha256(A)[:24]`` must not be selected by
    ``--tenant <that hash>`` when metadata/tasks identify tenant A.
    Hash-name fallback is only for dirs with no recorded identity (so a
    legacy tenant whose id equals ``sha256(other)[:24]`` is not deleted
    when cleaning ``other``). An unreadable hash sibling still matches A
    so cleanup can refuse the whole group instead of deleting legacy only.
    """
    if _is_mixed_tenant_dir(tdir):
        ids = set(_tenant_ids_from_tasks(tdir))
        meta = _tenant_id_from_meta(tdir)
        if meta is not None:
            ids.add(meta)
        if tenant in ids:
            return True
        return _unattributable_hash_dir(tdir) and tdir.name == _tenant_dir_key(
            tenant
        )
    recorded = _recorded_tenant_id(tdir)
    if recorded is not None:
        return recorded == tenant
    if tdir.name == _tenant_dir_key(tenant):
        return True
    return tdir.name == tenant


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
    """Return per-tenant task_count and age_days of newest task.

    Hashed and leftover legacy directories for the same original tenant_id
    are merged so operators see one row, unique task IDs, and the newest age.
    """
    now_ts = time.time() if now is None else now
    store_dir = store_dir.resolve()
    grouped: Dict[str, Dict[str, Any]] = {}
    for tdir in _tenant_dirs(store_dir):
        label = _tenant_label(tdir)
        stems = {p.stem for p in _task_files(tdir)}
        newest = _newest_task_mtime(tdir)
        mixed = _is_mixed_tenant_dir(tdir)
        row = grouped.get(label)
        if row is None:
            grouped[label] = {
                "task_ids": set(stems),
                "newest": newest,
                "mixed": mixed,
            }
            continue
        row["task_ids"].update(stems)
        row["mixed"] = bool(row.get("mixed")) or mixed
        if newest is not None and (
            row["newest"] is None or newest > row["newest"]
        ):
            row["newest"] = newest

    rows: List[Dict[str, Any]] = []
    for label in sorted(grouped):
        item = grouped[label]
        newest = item["newest"]
        age_days = None if newest is None else (now_ts - newest) / 86400.0
        rows.append(
            {
                "tenant": label,
                "task_count": len(item["task_ids"]),
                "age_days": age_days,
                "mixed": bool(item.get("mixed")),
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
                    "mixed": bool(r.get("mixed")),
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
        mixed_s = " mixed=true" if r.get("mixed") else ""
        print(
            f"tenant={r['tenant']} tasks={r['task_count']} "
            f"age_days={age_s}{mixed_s}"
        )
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
        tenants = [t for t in tenants if _tenant_matches(t, tenant)]
        if not tenants:
            print(f"tenant not found: {tenant}", file=sys.stderr)
            return 1

    groups: Dict[str, List[Path]] = {}
    for tdir in tenants:
        for ident in _dir_tenant_identities(tdir):
            bucket = groups.setdefault(ident, [])
            if tdir not in bucket:
                bucket.append(tdir)
    _attach_unreadable_hash_siblings(groups, tenants)

    removed = 0
    listed = 0
    refused = 0
    seen_paths = set()
    for label, dirs in groups.items():
        newest: Optional[float] = None
        for tdir in dirs:
            m = _newest_mtime(tdir)
            if m is not None and (newest is None or m > newest):
                newest = m
        if newest is None or newest > cutoff:
            continue
        age_days = (time.time() - newest) / 86400.0
        mixed_group = any(_is_mixed_tenant_dir(d) for d in dirs)
        for tdir in dirs:
            resolved = tdir.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            listed += 1
            if mixed_group:
                ids = list(_tenant_ids_from_tasks(tdir))
                meta = _tenant_id_from_meta(tdir)
                if meta is not None and meta not in ids:
                    ids = [meta] + ids
                print(
                    f"refused_mixed tenant={label} tenant_ids={','.join(ids)} "
                    f"age_days={age_days:.1f} path={tdir}",
                    file=sys.stderr,
                )
                refused += 1
                continue
            if dry_run:
                print(
                    f"would_delete tenant={label} age_days={age_days:.1f} "
                    f"path={tdir}"
                )
            else:
                shutil.rmtree(tdir)
                print(f"deleted tenant={label} age_days={age_days:.1f}")
                removed += 1

    mode = "dry_run" if dry_run else "apply"
    print(
        f"cleanup mode={mode} candidates={listed} deleted={removed} "
        f"refused_mixed={refused} older_than_days={older_than_days}"
    )
    return 1 if refused else 0


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
