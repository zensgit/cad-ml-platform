#!/usr/bin/env python3
"""Isolated-archive exercise: create ReviewReuseTask → export EvidencePack/audit.

Does NOT enable human decisions by default. Uses synthetic file bytes unless
--file is provided. Safe for offline CI demos (no live vision unless env set).

Examples::

  python scripts/review_reuse_isolated_archive_run.py --out /tmp/rr_export
  python scripts/review_reuse_isolated_archive_run.py --file sample.dxf
  python scripts/review_reuse_isolated_archive_run.py --file sample.dxf --seed-similar

Env (optional)::

  REVIEW_REUSE_STORE=filesystem
  REVIEW_REUSE_STORE_DIR=data/review_reuse_tasks
  REVIEW_REUSE_LIVE_DEDUP=false   # keep default-off for isolated samples
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_DEFAULT_IDEM = "isolated-archive-demo"
_LIVE_TRUE = frozenset({"1", "true", "yes", "on"})


def _generated_recall_mode(*, seed_similar: bool) -> str:
    """Seed vs live-recall must not share a generated idempotency key."""
    seed = "seed" if seed_similar else "unseeded"
    live = os.environ.get("REVIEW_REUSE_LIVE_DEDUP", "").strip().lower() in _LIVE_TRUE
    recall = "live" if live else "offline"
    return f"{seed}-{recall}"


def resolve_idempotency_key(
    explicit: str | None,
    file_bytes: bytes,
    *,
    from_file: bool,
    seed_similar: bool = False,
    file_name: str = "",
) -> str:
    """FILE runs must not reuse the synthetic demo key.

    Generated keys include basename and seed/recall mode so a later
    ``--seed-similar``, ``REVIEW_REUSE_LIVE_DEDUP`` change, or same-bytes
    different filename cannot silently replay or conflict.
    """
    if explicit:
        return explicit
    mode = _generated_recall_mode(seed_similar=seed_similar)
    if from_file:
        digest = hashlib.sha256(file_bytes).hexdigest()[:24]
        name_digest = hashlib.sha256(Path(file_name).name.encode("utf-8")).hexdigest()[
            :16
        ]
        return f"isolated-file-{mode}-{name_digest}-{digest}"
    return f"{_DEFAULT_IDEM}-{mode}"


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
        default=None,
        help="Create-task idempotency key (FILE default: name+content+mode)",
    )
    args = parser.parse_args(argv)

    # Local/dev identity posture for script usage.
    os.environ.setdefault("ENVIRONMENT", "development")
    # Never flip decision on from this script.
    os.environ.pop("REVIEW_REUSE_DECISIONS_ENABLED", None)

    from src.core.review_reuse.service import ReviewReuseError, ReviewReuseService
    from src.core.review_reuse.store import create_review_reuse_store

    store = create_review_reuse_store()
    svc = ReviewReuseService(store)

    if args.file is not None:
        if not args.file.is_file():
            print(f"error: --file not found: {args.file}", file=sys.stderr)
            return 2
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

    try:
        task = svc.create_task(
            tenant_id=args.tenant,
            file_name=file_name,
            file_bytes=file_bytes,
            idempotency_key=resolve_idempotency_key(
                args.idempotency_key,
                file_bytes,
                from_file=args.file is not None,
                seed_similar=args.seed_similar,
                file_name=file_name,
            ),
            seed_candidates=seed,
        )
    except ReviewReuseError as exc:
        print(f"error: {exc.code}: {exc.message}", file=sys.stderr)
        print("decisions=disabled (script never enables REVIEW_REUSE_DECISIONS_ENABLED)")
        return 2
    try:
        pack, md = svc.get_evidence_pack(args.tenant, task.task_id, as_markdown=True)
        audit = svc.export_audit_bundle(args.tenant, task.task_id)
    except ReviewReuseError as exc:
        # Overlap or a fresh crash returns the running snapshot with no pack.
        print(f"error: {exc.code}: {exc.message}", file=sys.stderr)
        print(f"task_id={task.task_id}")
        print(f"status={task.status.value}")
        print("decisions=disabled (script never enables REVIEW_REUSE_DECISIONS_ENABLED)")
        return 2

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
