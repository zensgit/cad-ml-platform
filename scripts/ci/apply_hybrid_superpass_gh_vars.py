#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from typing import Dict, List, Tuple


def _build_var_map(config_path: str) -> Dict[str, str]:
    config_text = str(config_path).strip()
    if not config_text:
        raise ValueError("--config-path must not be empty")
    return {
        "HYBRID_SUPERPASS_ENABLE": "true",
        "HYBRID_SUPERPASS_MISSING_MODE": "fail",
        "HYBRID_SUPERPASS_FAIL_ON_FAILED": "true",
        "HYBRID_SUPERPASS_VALIDATION_STRICT": "false",
        "HYBRID_SUPERPASS_VALIDATION_SCHEMA_MODE": "builtin",
        "HYBRID_SUPERPASS_CONFIG": config_text,
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
        description="Apply hybrid superpass baseline to GitHub variables."
    )
    parser.add_argument("--repo", required=True, help="GitHub repo, e.g. owner/repo")
    parser.add_argument(
        "--config-path",
        default="config/hybrid_superpass_targets.yaml",
        help="HYBRID_SUPERPASS_CONFIG value",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    var_map = _build_var_map(args.config_path)
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
