#!/usr/bin/env python3
"""Isolated-archive exercise: create ReviewReuseTask → export EvidencePack/audit.

Does NOT enable human decisions by default. Uses synthetic file bytes unless
--file is provided. Safe for offline CI demos (no live vision unless env set).

Examples::

  python scripts/review_reuse_isolated_archive_run.py --out /tmp/rr_export
  python scripts/review_reuse_isolated_archive_run.py --file sample.dxf --seed-similar

Env (optional)::

  REVIEW_REUSE_STORE=filesystem
  REVIEW_REUSE_STORE_DIR=data/review_reuse_tasks
  REVIEW_REUSE_LIVE_DEDUP=false   # keep default-off for isolated samples
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/isolated_samples/synthetic_run/exports"),
        help="Directory for EvidencePack + audit export",
    )
    parser.add_argument("--file", type=Path, default=None, help="Optional drawing file")
    parser.add_argument(
        "--tenant",
        default="isolated-sample",
        help="Tenant id for this isolated run",
    )
    parser.add_argument(
        "--seed-similar",
        action="store_true",
        help="Attach a synthetic similar candidate (offline archive fixture)",
    )
    parser.add_argument(
        "--idempotency-key",
        default="isolated-archive-demo",
        help="Create-task idempotency key",
    )
    args = parser.parse_args(argv)

    # Local/dev identity posture for script usage.
    os.environ.setdefault("ENVIRONMENT", "development")
    # Never flip decision on from this script.
    os.environ.pop("REVIEW_REUSE_DECISIONS_ENABLED", None)

    from src.core.review_reuse.service import ReviewReuseService
    from src.core.review_reuse.store import create_review_reuse_store

    store = create_review_reuse_store()
    svc = ReviewReuseService(store)

    if args.file is not None:
        file_bytes = args.file.read_bytes()
        file_name = args.file.name
    else:
        file_bytes = b"0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nEOF\n"
        file_name = "synthetic_isolated.dxf"

    seed = None
    if args.seed_similar:
        seed = [
            {
                "candidate_id": "synthetic-archive-001",
                "candidate_source": "archive",
                "state": "similar",
                "scores": {"geometric": 0.88, "semantic": 0.70},
                "verification": {
                    "verdict": "similar",
                    "level": 2,
                    "methods": ["synthetic-fixture"],
                },
            }
        ]

    task = svc.create_task(
        tenant_id=args.tenant,
        file_name=file_name,
        file_bytes=file_bytes,
        idempotency_key=args.idempotency_key,
        seed_candidates=seed,
    )
    pack, md = svc.get_evidence_pack(args.tenant, task.task_id, as_markdown=True)
    audit = svc.export_audit_bundle(args.tenant, task.task_id)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "task.json").write_text(
        json.dumps(task.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    (args.out / "evidence.json").write_text(
        json.dumps(pack, indent=2), encoding="utf-8"
    )
    (args.out / "evidence.md").write_text(md or "", encoding="utf-8")
    (args.out / "audit_bundle.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    print(f"task_id={task.task_id}")
    print(f"status={task.status.value}")
    print(f"candidates={len(task.candidates)}")
    print(f"exports={args.out.resolve()}")
    print("decisions=disabled (script never enables REVIEW_REUSE_DECISIONS_ENABLED)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
