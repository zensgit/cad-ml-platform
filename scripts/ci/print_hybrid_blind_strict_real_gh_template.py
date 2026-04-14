#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import List


def build_workflow_run_command(
    *,
    workflow: str,
    ref: str,
    repo: str,
    dxf_dir: str,
) -> List[str]:
    command = ["gh", "workflow", "run", workflow, "--ref", ref]
    if str(repo).strip():
        command.extend(["--repo", str(repo).strip()])
    command.extend(
        [
            "-f",
            "hybrid_blind_enable=true",
            "-f",
            f"hybrid_blind_dxf_dir={str(dxf_dir).strip()}",
            "-f",
            "hybrid_blind_fail_on_gate_failed=true",
            "-f",
            "hybrid_blind_strict_require_real_data=true",
        ]
    )
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print a GitHub CLI template for strict-real hybrid blind runs."
    )
    parser.add_argument("--workflow", default="evaluation-report.yml")
    parser.add_argument("--ref", default="main")
    parser.add_argument("--repo", default="")
    parser.add_argument("--dxf-dir", required=True)
    parser.add_argument("--print-vars", action="store_true")
    parser.add_argument("--print-watch", action="store_true")
    args = parser.parse_args(argv)

    command = build_workflow_run_command(
        workflow=str(args.workflow),
        ref=str(args.ref),
        repo=str(args.repo),
        dxf_dir=str(args.dxf_dir),
    )
    print(" ".join(command))

    if bool(args.print_vars):
        print("HYBRID_BLIND_ENABLE=true")
        print(f"HYBRID_BLIND_DXF_DIR={str(args.dxf_dir).strip()}")
        print("HYBRID_BLIND_FAIL_ON_GATE_FAILED=true")
        print("HYBRID_BLIND_STRICT_REQUIRE_REAL_DATA=true")

    if bool(args.print_watch):
        print("gh run watch <run_id> --exit-status")
        print("gh run view <run_id> --json conclusion,url")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
