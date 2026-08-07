#!/usr/bin/env python3
"""Rewrite workflow ``uses: owner/repo@tag`` lines to commit SHAs.

Used with Dependabot github-actions PRs so Action Pin Guard can pass.
SHA resolution uses the GitHub REST API (stdlib only). Already-SHA refs
are left unchanged. Local / docker / reusable workflow paths (``./``) are
skipped.

After rewriting, callers should regenerate
``config/workflow_action_pin_policy.json`` via
``generate_workflow_action_pin_policy.py`` so new SHAs are allow-listed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

_USES_RE = re.compile(r"^(\s*(?:-\s*)?uses:\s*)([^\s#]+)(.*)$")
_HEX40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_ACTION_RE = re.compile(
    r"^(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)(?P<path>/[A-Za-z0-9_./-]*)?$"
)


def _iter_workflow_files(workflows_dir: Path) -> List[Path]:
    if not workflows_dir.exists() or not workflows_dir.is_dir():
        return []
    return sorted(path for path in workflows_dir.glob("*.yml") if path.is_file())


def _parse_uses_target(token: str) -> Tuple[str, str]:
    token = (token or "").strip()
    if "@" not in token:
        return ("", "")
    action, ref = token.split("@", 1)
    return (action.strip(), ref.strip())


def resolve_ref_to_sha(
    *,
    owner: str,
    repo: str,
    ref: str,
    token: str = "",
    opener: Optional[Callable[..., object]] = None,
) -> str:
    """Return a 40-char lowercase SHA for ``owner/repo@ref``, or empty on failure."""

    if _HEX40_RE.fullmatch(ref):
        return ref.lower()

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "cad-ml-platform-action-pin-fixer",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Prefer annotated/lightweight tags, then heads, then commit-ish.
    candidates = [
        f"https://api.github.com/repos/{owner}/{repo}/git/ref/tags/{ref}",
        f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/{ref}",
        f"https://api.github.com/repos/{owner}/{repo}/commits/{ref}",
    ]

    open_url = opener or urllib.request.urlopen
    for url in candidates:
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with open_url(req, timeout=30) as resp:  # type: ignore[arg-type]
                body = resp.read().decode("utf-8")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
            continue
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            continue
        sha = ""
        if isinstance(payload, dict):
            obj = payload.get("object")
            if isinstance(obj, dict) and obj.get("type") == "tag":
                # Annotated tag → resolve tag object to commit.
                tag_url = str(obj.get("url") or "")
                if tag_url:
                    treq = urllib.request.Request(tag_url, headers=headers, method="GET")
                    try:
                        with open_url(treq, timeout=30) as tresp:  # type: ignore[arg-type]
                            tbody = tresp.read().decode("utf-8")
                        tobj = json.loads(tbody)
                        if isinstance(tobj, dict):
                            nested = tobj.get("object")
                            if isinstance(nested, dict):
                                sha = str(nested.get("sha") or "")
                    except (
                        urllib.error.HTTPError,
                        urllib.error.URLError,
                        TimeoutError,
                        OSError,
                        json.JSONDecodeError,
                    ):
                        sha = ""
            if not sha and isinstance(obj, dict):
                sha = str(obj.get("sha") or "")
            if not sha:
                sha = str(payload.get("sha") or "")
        if _HEX40_RE.fullmatch(sha):
            return sha.lower()
    return ""


def rewrite_workflow_text(
    text: str,
    *,
    resolve: Callable[[str, str, str], str],
) -> Tuple[str, List[Dict[str, str]]]:
    """Rewrite non-SHA uses targets. ``resolve(owner, repo, ref) -> sha``."""

    changes: List[Dict[str, str]] = []
    out_lines: List[str] = []
    for line in text.splitlines(keepends=True):
        raw = line.rstrip("\n")
        newline = "\n" if line.endswith("\n") else ""
        match = _USES_RE.match(raw)
        if not match:
            out_lines.append(line)
            continue
        prefix, target, suffix = match.group(1), match.group(2), match.group(3)
        if target.startswith("./") or target.startswith("docker://"):
            out_lines.append(line)
            continue
        action, ref = _parse_uses_target(target)
        if not action or not ref:
            out_lines.append(line)
            continue
        if _HEX40_RE.fullmatch(ref):
            out_lines.append(line)
            continue
        am = _ACTION_RE.match(action)
        if not am:
            out_lines.append(line)
            continue
        owner = am.group("owner")
        repo = am.group("repo")
        sha = resolve(owner, repo, ref)
        if not sha:
            out_lines.append(line)
            changes.append(
                {
                    "status": "unresolved",
                    "uses": target,
                    "action": action,
                    "ref": ref,
                }
            )
            continue
        new_target = f"{action}@{sha}"
        # Keep any existing trailing comment; otherwise record the original tag/ref.
        if "#" in suffix:
            trailing = suffix
        else:
            trailing = f"  # {ref}"
        new_line = f"{prefix}{new_target}{trailing}{newline}"
        out_lines.append(new_line)
        changes.append(
            {
                "status": "rewritten",
                "uses": target,
                "new_uses": new_target,
                "action": action,
                "ref": ref,
                "sha": sha,
            }
        )
    return ("".join(out_lines), changes)


def rewrite_workflows_dir(
    workflows_dir: Path,
    *,
    token: str = "",
    dry_run: bool = False,
    resolve_fn: Optional[Callable[[str, str, str], str]] = None,
) -> Dict[str, object]:
    def _default_resolve(owner: str, repo: str, ref: str) -> str:
        return resolve_ref_to_sha(owner=owner, repo=repo, ref=ref, token=token)

    resolve = resolve_fn or _default_resolve
    all_changes: List[Dict[str, str]] = []
    files_rewritten = 0
    for path in _iter_workflow_files(workflows_dir):
        original = path.read_text(encoding="utf-8")
        rewritten, changes = rewrite_workflow_text(original, resolve=resolve)
        if changes:
            all_changes.extend({**c, "file": str(path)} for c in changes)
        if rewritten != original:
            files_rewritten += 1
            if not dry_run:
                path.write_text(rewritten, encoding="utf-8")
    unresolved = [c for c in all_changes if c.get("status") == "unresolved"]
    return {
        "status": "ok" if not unresolved else "partial",
        "workflows_dir": str(workflows_dir),
        "files_rewritten": files_rewritten,
        "changes": all_changes,
        "unresolved_count": len(unresolved),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite workflow uses: tags/branches to commit SHAs."
    )
    parser.add_argument("--workflows-dir", default=".github/workflows")
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN", "") or os.environ.get("GH_TOKEN", ""),
        help="GitHub token for API resolution (optional for public repos).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args(argv)

    report = rewrite_workflows_dir(
        Path(str(args.workflows_dir)).expanduser(),
        token=str(args.token or ""),
        dry_run=bool(args.dry_run),
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        out = Path(str(args.output_json)).expanduser()
        if out.parent != Path("."):
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"{text}\n", encoding="utf-8")
    # Non-zero only when a non-SHA ref could not be resolved (fail closed for CI use).
    return 1 if int(report.get("unresolved_count") or 0) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
