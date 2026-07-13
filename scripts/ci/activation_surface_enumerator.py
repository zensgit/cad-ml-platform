#!/usr/bin/env python3
"""L3 activation-surface enumerator — completeness by construction, not by a hand-list.

Every place that deserializes model bytes into a process (``torch.load`` /
``pickle.load`` / ``pickle.loads`` / ``joblib.load`` / ``*.load_state_dict`` / a ``reload_model(``
call) is a potential model-ACTIVATION surface. A hand-maintained count has been wrong repeatedly, so
this enumerator inverts the burden: it discovers every load site via the AST and requires each to be
CLASSIFIED in ``scripts/ci/activation_surface.json``. A NEW, un-annotated load site fails CI RED —
so a new activation surface cannot land silently and must be classified (and, once the L3 proof
membrane exists, ``gated`` sites must route through ``verify_and_load``).

Classes (per the L3 design-lock §1):
  gated     — production-reachable activation; MUST pass the proof membrane (raises today, unbuilt)
  producer  — emits an artifact offline; does not activate a running service
  offline   — a CLI/tool load that mutates no running production service
  unmounted — a scaffold reachable by 0 mounted routes today (auto-promotes to `gated` if mounted)
  infra     — a non-model deserialization (calibrator / vector store / cache) — not a classifier

This is DISCOVERY + fail-closed bookkeeping only. It changes no runtime behaviour and can never emit
a "green that enables" — it only ever passes (all sites classified) or reds (an unclassified site).

Exit 0 = every load site is classified and every manifest entry still resolves.
Exit 1 = a new unclassified load site, or a stale manifest entry. Stdlib only.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "scripts" / "ci" / "activation_surface.json"
SCAN_DIRS = ("src", "scripts")
VALID_CLASSES = {"gated", "producer", "offline", "unmounted", "infra"}

# Call targets that deserialize bytes into a live object graph (arbitrary-code-exec on load for
# torch.load(weights_only=False) / pickle / joblib). load_state_dict loads weights INTO a model.
_LOAD_ATTRS = {"load", "loads", "load_state_dict"}
_LOAD_ATTR_ROOTS = {"torch", "pickle", "joblib"}          # torch.load / pickle.load(s) / joblib.load
_LOAD_CALL_NAMES = {"reload_model"}                        # bare reload_model( call


def _qualname(stack: List[str]) -> str:
    return ".".join(stack) if stack else "<module>"


class _LoadVisitor(ast.NodeVisitor):
    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        self.stack: List[str] = []
        self.sites: List[Tuple[str, str, int]] = []  # (qualname, kind, lineno)

    def _scoped(self, name: str, node: ast.AST) -> None:
        self.stack.append(name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scoped(node.name, node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scoped(node.name, node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scoped(node.name, node)

    def visit_Call(self, node: ast.Call) -> None:
        kind = self._classify_call(node.func)
        if kind:
            self.sites.append((_qualname(self.stack), kind, node.lineno))
        self.generic_visit(node)

    def _classify_call(self, func: ast.AST):
        # torch.load / pickle.load / pickle.loads / joblib.load / <x>.load_state_dict
        if isinstance(func, ast.Attribute) and func.attr in _LOAD_ATTRS:
            if func.attr == "load_state_dict":
                return "load_state_dict"
            root = func.value
            if isinstance(root, ast.Name) and root.id in _LOAD_ATTR_ROOTS:
                return f"{root.id}.{func.attr}"
        # bare reload_model(
        if isinstance(func, ast.Name) and func.id in _LOAD_CALL_NAMES:
            return f"{func.id}()"
        return None


def enumerate_sites() -> Dict[str, dict]:
    """Return {key: {file, symbol, kind, lineno}} for every discovered load site.

    key = "<relpath>::<qualname>::<kind>#<n>" — stable across line-number drift (keyed on the
    enclosing symbol + call kind, not the line), so reformatting does not spuriously red CI.
    """
    found: Dict[str, dict] = {}
    for d in SCAN_DIRS:
        for py in sorted((REPO_ROOT / d).rglob("*.py")):
            rel = py.relative_to(REPO_ROOT).as_posix()
            if "/tests/" in f"/{rel}" or rel.startswith("tests/") or py.name.startswith("test_"):
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"), filename=rel)
            except (SyntaxError, UnicodeDecodeError):
                continue
            v = _LoadVisitor(rel)
            v.visit(tree)
            per_key_count: Dict[str, int] = {}
            for qual, kind, lineno in v.sites:
                base = f"{rel}::{qual}::{kind}"
                n = per_key_count.get(base, 0)
                per_key_count[base] = n + 1
                found[f"{base}#{n}"] = {"file": rel, "symbol": qual, "kind": kind, "lineno": lineno}
    return found


def load_manifest() -> Dict[str, dict]:
    if not MANIFEST.exists():
        raise SystemExit(f"[activation-enumerator] manifest missing: {MANIFEST}")
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = data.get("sites", {})
    for key, entry in entries.items():
        cls = entry.get("class")
        if cls not in VALID_CLASSES:
            raise SystemExit(
                f"[activation-enumerator] manifest entry {key!r} has invalid class {cls!r} "
                f"(valid: {sorted(VALID_CLASSES)})"
            )
    return entries


def main(argv=None) -> int:
    found = enumerate_sites()
    manifest = load_manifest()
    found_keys, manifest_keys = set(found), set(manifest)

    unclassified = sorted(found_keys - manifest_keys)
    stale = sorted(manifest_keys - found_keys)

    if unclassified:
        sys.stderr.write(
            "[activation-enumerator] RED — new UNCLASSIFIED model-load site(s); a new activation "
            "surface must be classified in scripts/ci/activation_surface.json before it can land:\n"
        )
        for k in unclassified:
            s = found[k]
            sys.stderr.write(f"  + {s['file']}:{s['lineno']}  {s['kind']}  in {s['symbol']}   key={k}\n")
    if stale:
        sys.stderr.write(
            "[activation-enumerator] RED — STALE manifest entr(y/ies) no longer resolve to a load "
            "site (re-key or remove them):\n"
        )
        for k in stale:
            sys.stderr.write(f"  - {k}  (was: {manifest[k].get('class')})\n")

    if unclassified or stale:
        return 1

    by_class: Dict[str, int] = {}
    for entry in manifest.values():
        by_class[entry["class"]] = by_class.get(entry["class"], 0) + 1
    summary = ", ".join(f"{c}={by_class.get(c, 0)}" for c in sorted(VALID_CLASSES))
    print(f"[activation-enumerator] OK — {len(found_keys)} load sites, all classified ({summary}).")
    gated = [k for k, e in manifest.items() if e["class"] == "gated"]
    print(f"[activation-enumerator] {len(gated)} `gated` production-reachable activation point(s); "
          "each MUST route through the L3 proof membrane (verify_and_load) once it exists — until "
          "then the membrane default is #509's unconditional raise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
