#!/usr/bin/env python3
"""Export ReviewReuse audit_bundle.json + evidence.md for an existing task.

Loads a task by --tenant and --task-id from the configured store (memory or
filesystem via REVIEW_REUSE_STORE / REVIEW_REUSE_STORE_DIR), then writes a
quarantined audit export under --out.

Does NOT enable human decisions. Export is audit_quarantine only (R2 HOLD —
not a training-readable path).

Examples::

  REVIEW_REUSE_STORE=filesystem REVIEW_REUSE_STORE_DIR=data/review_reuse_tasks \\
    python scripts/review_reuse_export_audit.py \\
      --tenant my-tenant --task-id <uuid> --out /tmp/rr_audit

  make review-reuse-export-audit TENANT=my-tenant TASK_ID=<uuid> OUT=/tmp/rr_audit

Env (optional)::

  REVIEW_REUSE_STORE=filesystem|memory   (default: memory)
  REVIEW_REUSE_STORE_DIR=data/review_reuse_tasks
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tenant",
        required=True,
        help="Tenant id that owns the task",
    )
    parser.add_argument(
        "--task-id",
        required=True,
        dest="task_id",
        help="Task id to export",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for audit_bundle.json and evidence.md",
    )
    args = parser.parse_args(argv)

    # Local/dev identity posture for script usage.
    os.environ.setdefault("ENVIRONMENT", "development")
    # Never flip decision on from this script (R2 / pilot fail-closed).
    os.environ.pop("REVIEW_REUSE_DECISIONS_ENABLED", None)

    from src.core.review_reuse.service import ReviewReuseError, ReviewReuseService
    from src.core.review_reuse.store import create_review_reuse_store

    store = create_review_reuse_store()
    svc = ReviewReuseService(store)

    try:
        audit = svc.export_audit_bundle(args.tenant, args.task_id)
    except ReviewReuseError as exc:
        print(f"error={exc.code}: {exc.message}", file=sys.stderr)
        return 1

    md = audit.get("evidence_pack_markdown") or ""

    args.out.mkdir(parents=True, exist_ok=True)
    audit_path = args.out / "audit_bundle.json"
    evidence_md_path = args.out / "evidence.md"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    evidence_md_path.write_text(md, encoding="utf-8")

    print(f"task_id={args.task_id}")
    print(f"tenant={args.tenant}")
    print(f"export_kind={audit.get('export_kind')}")
    print(f"schema_version={audit.get('schema_version')}")
    print(f"audit_bundle={audit_path.resolve()}")
    print(f"evidence_md={evidence_md_path.resolve()}")
    print("decisions=disabled (script never enables REVIEW_REUSE_DECISIONS_ENABLED)")
    print("r2_hold=audit_quarantine (not training-readable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
