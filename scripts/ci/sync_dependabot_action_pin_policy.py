#!/usr/bin/env python3
"""Dependabot follow-up for Action Pin Guard (#476).

1. Rewrite any non-SHA ``uses:`` refs in workflows to commit SHAs.
2. Regenerate ``config/workflow_action_pin_policy.json`` from the resulting
   workflow tree so new SHAs are allow-listed.

Dependabot already often emits commit SHAs for github-actions bumps; those
still fail Action Pin Guard when the *policy* only lists the previous SHA.
Step 2 is therefore the load-bearing fix; step 1 covers remaining tag bumps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as a script from any CWD.
_SCRIPTS_CI = Path(__file__).resolve().parent
if str(_SCRIPTS_CI) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_CI))

from generate_workflow_action_pin_policy import (  # noqa: E402
    collect_action_pin_policy,
)
from rewrite_workflow_action_tags_to_sha import (  # noqa: E402
    rewrite_workflows_dir,
)


def sync_policy(
    *,
    workflows_dir: Path,
    policy_json: Path,
    token: str = "",
    dry_run: bool = False,
    skip_rewrite: bool = False,
) -> Dict[str, Any]:
    rewrite_report: Dict[str, Any]
    if skip_rewrite:
        rewrite_report = {
            "status": "skipped",
            "files_rewritten": 0,
            "changes": [],
            "unresolved_count": 0,
        }
    else:
        rewrite_report = rewrite_workflows_dir(
            workflows_dir, token=token, dry_run=dry_run
        )

    payload, exit_code = collect_action_pin_policy(
        workflows_dir=workflows_dir, strict=True
    )
    if not dry_run:
        if policy_json.parent != Path("."):
            policy_json.parent.mkdir(parents=True, exist_ok=True)
        policy_json.write_text(
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n",
            encoding="utf-8",
        )

    return {
        "status": "ok"
        if exit_code == 0 and int(rewrite_report.get("unresolved_count") or 0) == 0
        else "error",
        "rewrite": rewrite_report,
        "policy": {
            "path": str(policy_json),
            "actions_count": len(payload.get("actions") or {}),
            "non_sha_refs": payload.get("non_sha_refs") or [],
            "strict_exit_code": exit_code,
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite action tags to SHAs and regenerate pin policy (#476)."
    )
    parser.add_argument("--workflows-dir", default=".github/workflows")
    parser.add_argument(
        "--policy-json", default="config/workflow_action_pin_policy.json"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN", "") or os.environ.get("GH_TOKEN", ""),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-rewrite",
        action="store_true",
        help="Only regenerate policy (workflows already SHA-pinned).",
    )
    parser.add_argument("--output-json", default="")
    args = parser.parse_args(argv)

    report = sync_policy(
        workflows_dir=Path(str(args.workflows_dir)).expanduser(),
        policy_json=Path(str(args.policy_json)).expanduser(),
        token=str(args.token or ""),
        dry_run=bool(args.dry_run),
        skip_rewrite=bool(args.skip_rewrite),
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        out = Path(str(args.output_json)).expanduser()
        if out.parent != Path("."):
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"{text}\n", encoding="utf-8")
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
