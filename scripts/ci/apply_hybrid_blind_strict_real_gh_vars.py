#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from typing import Dict, List, Tuple


def _build_var_map(dxf_dir: str) -> Dict[str, str]:
    path_text = str(dxf_dir).strip()
    if not path_text:
        raise ValueError("--dxf-dir must not be empty")
    return {
        "HYBRID_BLIND_DXF_DIR": path_text,
        "HYBRID_BLIND_DRIFT_ALERT_ENABLE": "true",
        "HYBRID_BLIND_ENABLE": "true",
        "HYBRID_BLIND_FAIL_ON_GATE_FAILED": "true",
        "HYBRID_BLIND_STRICT_REQUIRE_REAL_DATA": "true",
    }


def _print_plan(repo: str, var_map: Dict[str, str]) -> None:
    print("plan=gh_variables")
    print(f"repo={repo}")
    for key in sorted(var_map.keys()):
        print(f"{key}={var_map[key]}")


def _apply(repo: str, var_map: Dict[str, str]) -> List[Tuple[str, int, str]]:
    results: List[Tuple[str, int, str]] = []
    for key in sorted(var_map.keys()):
        value = var_map[key]
        proc = subprocess.run(
            ["gh", "variable", "set", key, "--repo", repo, "--body", value],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (proc.stdout or proc.stderr or "").strip()
        results.append((key, int(proc.returncode), out))
    return results


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply strict-real hybrid blind baseline to GitHub variables."
    )
    parser.add_argument("--repo", required=True, help="GitHub repo, e.g. owner/repo")
    parser.add_argument(
        "--dxf-dir",
        required=True,
        help="Real DXF directory used by strict-real hybrid blind workflow runs.",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    var_map = _build_var_map(args.dxf_dir)
    _print_plan(args.repo, var_map)

    if not args.apply:
        print("apply=false")
        return 0

    results = _apply(args.repo, var_map)
    failed = [row for row in results if row[1] != 0]
    for key, code, message in results:
        status = "ok" if code == 0 else "failed"
        print(f"result {key} status={status} code={code} message={message}")
    print(f"applied={len(results)} failed={len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
