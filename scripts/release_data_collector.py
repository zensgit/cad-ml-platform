#!/usr/bin/env python3
"""
Release Data Collector
Collects change, test, dependency, error-code, and metrics deltas between base branch and HEAD.

This is a lean implementation designed to support the release risk scorer workflow.

Inputs:
- --base-branch: base branch name (e.g., main)
- --output: output JSON filepath (default: - for stdout)

Outputs JSON with keys:
- git: { base_sha, compare_range, files_changed, lines_added, lines_deleted, by_area }
- tests: { total, passed, failed, errors, skipped }
- deps:  { added, removed }
- errors: { added, removed, total_current, total_base }
- metrics: { added, removed, total_current, total_base }

Assumptions:
- Git history is fully fetched (fetch-depth: 0)
- Base branch exists locally or as origin/<base>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def run(cmd: List[str], cwd: Optional[str] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def resolve_base(base_branch: str) -> Tuple[str, str]:
    candidates = [base_branch, f"origin/{base_branch}"]
    for ref in candidates:
        try:
            sha = run(["git", "rev-parse", ref]).stdout.strip()
            if sha:
                merge_base = run(["git", "merge-base", "HEAD", ref]).stdout.strip()
                return ref, merge_base or sha
        except subprocess.CalledProcessError:
            continue
    raise RuntimeError(f"Cannot resolve base branch: {base_branch}")


def git_diff_stats(base: str) -> Tuple[int, int, int, Dict[str, int]]:
    # files_changed, lines_added, lines_deleted
    name_status = run(["git", "diff", "--name-status", f"{base}..HEAD"]).stdout.splitlines()
    numstat = run(["git", "diff", "--numstat", f"{base}..HEAD"]).stdout.splitlines()

    files_changed = len(name_status)
    added = 0
    deleted = 0
    areas = {"src": 0, "tests": 0, "scripts": 0, "docs": 0, "workflows": 0, "other": 0}
    for line in numstat:
        parts = line.split()  # e.g., '12\t5\tpath'
        if len(parts) < 3:
            continue
        try:
            a, d = int(parts[0]), int(parts[1])
        except ValueError:
            # binary file changes are represented as -
            a, d = 0, 0
        added += a
        deleted += d
        path = parts[2]
        if path.startswith("src/"):
            areas["src"] += 1
        elif path.startswith("tests/"):
            areas["tests"] += 1
        elif path.startswith("scripts/"):
            areas["scripts"] += 1
        elif path.startswith("docs/"):
            areas["docs"] += 1
        elif path.startswith(".github/workflows/"):
            areas["workflows"] += 1
        else:
            areas["other"] += 1
    return files_changed, added, deleted, areas


def load_file_from_ref(ref: str, path: str) -> str:
    try:
        return run(["git", "show", f"{ref}:{path}"], check=True).stdout
    except subprocess.CalledProcessError:
        return ""


def dependency_diff(base_ref: str) -> Tuple[int, int]:
    cur = Path("requirements.txt").read_text(encoding="utf-8", errors="ignore") if Path("requirements.txt").exists() else ""
    base = load_file_from_ref(base_ref, "requirements.txt")
    cur_set = set([l.strip() for l in cur.splitlines() if l.strip() and not l.startswith("#")])
    base_set = set([l.strip() for l in base.splitlines() if l.strip() and not l.startswith("#")])
    added = len(cur_set - base_set)
    removed = len(base_set - cur_set)
    return added, removed


ENUM_RE = re.compile(r"class\s+ErrorCode\(.*?\):([\s\S]*?)class|class\s+ErrorCode\(.*?\):([\s\S]*)", re.MULTILINE)
MEMBER_RE = re.compile(r"^\s*[A-Z0-9_]+\s*=\s*\"[A-Za-z0-9_]+\"", re.MULTILINE)


def parse_error_codes(text: str) -> List[str]:
    if not text:
        return []
    m = ENUM_RE.search(text)
    if not m:
        # fallback: collect top-level members
        return [m.group(0).strip().split("=")[0].strip() for m in MEMBER_RE.finditer(text)]
    body = m.group(1) or m.group(2) or ""
    return [m.group(0).strip().split("=")[0].strip() for m in MEMBER_RE.finditer(body)]


def error_code_diff(base_ref: str) -> Tuple[int, int, int, int]:
    files = [
        "src/core/errors.py",
        "src/core/errors_extended.py",
    ]
    cur_set = set()
    base_set = set()
    for f in files:
        cur_text = Path(f).read_text(encoding="utf-8", errors="ignore") if Path(f).exists() else ""
        base_text = load_file_from_ref(base_ref, f)
        cur_set.update(parse_error_codes(cur_text))
        base_set.update(parse_error_codes(base_text))
    added = len(cur_set - base_set)
    removed = len(base_set - cur_set)
    return added, removed, len(cur_set), len(base_set)


METRIC_NAME_RE = re.compile(r"^(?:\s*)[A-Za-z_][A-Za-z0-9_]*\s*=\s*(Counter|Gauge|Histogram)\(\s*['\"]([a-zA-Z_:][a-zA-Z0-9_:]*)['\"]", re.MULTILINE)


def metric_names_from_text(text: str) -> List[str]:
    return [m.group(2) for m in METRIC_NAME_RE.finditer(text or "")]


def metrics_diff(base_ref: str) -> Tuple[int, int, int, int]:
    files = [
        "src/utils/metrics.py",
        "src/core/resilience/adaptive_rate_limiter.py",
    ]
    cur_set = set()
    base_set = set()
    for f in files:
        cur_text = Path(f).read_text(encoding="utf-8", errors="ignore") if Path(f).exists() else ""
        base_text = load_file_from_ref(base_ref, f)
        cur_set.update(metric_names_from_text(cur_text))
        base_set.update(metric_names_from_text(base_text))
    added = len(cur_set - base_set)
    removed = len(base_set - cur_set)
    return added, removed, len(cur_set), len(base_set)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect release risk related data")
    ap.add_argument("--base-branch", required=True)
    ap.add_argument("--output", default="-")
    args = ap.parse_args()

    ref, base_sha = resolve_base(args.base_branch)
    files_changed, added, deleted, areas = git_diff_stats(base_sha)

    # Tests from environment (populated by CI step)
    tests = {
        "total": int(os.getenv("TEST_TOTAL", "0")),
        "passed": int(os.getenv("TEST_PASSED", "0")),
        "failed": int(os.getenv("TEST_FAILED", "0")),
        "errors": int(os.getenv("TEST_ERRORS", "0")),
        "skipped": int(os.getenv("TEST_SKIPPED", "0")),
    }

    deps_added, deps_removed = dependency_diff(ref)
    ec_added, ec_removed, ec_cur, ec_base = error_code_diff(ref)
    m_added, m_removed, m_cur, m_base = metrics_diff(ref)

    payload = {
        "git": {
            "base_ref": ref,
            "base_sha": base_sha,
            "compare_range": f"{base_sha}..HEAD",
            "files_changed": files_changed,
            "lines_added": added,
            "lines_deleted": deleted,
            "by_area": areas,
        },
        "tests": tests,
        "deps": {"added": deps_added, "removed": deps_removed},
        "errors": {
            "added": ec_added,
            "removed": ec_removed,
            "total_current": ec_cur,
            "total_base": ec_base,
        },
        "metrics": {
            "added": m_added,
            "removed": m_removed,
            "total_current": m_cur,
            "total_base": m_base,
        },
    }

    out = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output == "-":
        print(out)
    else:
        Path(args.output).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()

