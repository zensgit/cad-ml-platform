#!/usr/bin/env python3
"""L3 activation-surface enumerator — completeness by construction, not by a hand-list.

Discovers every place that deserializes model bytes into a process and requires each to be
CLASSIFIED in ``scripts/ci/activation_surface.json``. A NEW, un-annotated load site fails CI RED.

**Import-aware** (review 5 — a name-only matcher had real blind spots): resolves import aliases so
``import torch as t; t.load(...)`` and ``from torch import load; load(...)`` and
``import pickle as p; p.loads(...)`` are all detected, and covers the model-loading patterns a
CAD/OCR platform actually uses beyond raw ``torch.load``:

  * ``torch.load`` / ``pickle.load`` / ``pickle.loads`` / ``joblib.load`` — via the resolved module,
    NOT the literal spelling (alias- and from-import-aware);
  * ``*.load_state_dict(...)`` — loads weights into a live model;
  * ``*.from_pretrained(...)`` — the HuggingFace/transformers weight loader (e.g. DeepSeek OCR);
  * a curated set of model-constructor calls that load weights on construction
    (``SentenceTransformer`` / ``CrossEncoder`` / ``PaddleOCR`` / …), import-aware;
  * a ``reload_model(`` call.

Classes (per the L3 design-lock §1): ``gated`` (production-reachable activation; MUST pass the proof
membrane / be frozen), ``producer`` (offline artifact emitter), ``offline`` (a CLI/tool load),
``unmounted`` (0-route scaffold; auto-promotes to gated if mounted), ``infra`` (non-model
deserialization — calibrator / vector-store / cache).

Discovery + fail-closed bookkeeping ONLY: it can never emit a "green that enables" — it only passes
(all classified) or reds. Not provably exhaustive of every possible Python model load; it covers the
enumerated patterns above and reds on any NEW site matching them. Widen the patterns as new loader
idioms appear. Exit 0 = all classified + no stale entry; exit 1 otherwise. Stdlib only.
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

# Deserializer modules and their load attributes.
_MODULE_LOADERS = {"torch": {"load"}, "pickle": {"load", "loads"}, "joblib": {"load"}}
# Model constructors that load weights on construction (import-aware; curated — extend as needed).
_MODEL_CONSTRUCTORS = {"SentenceTransformer", "CrossEncoder", "PaddleOCR"}
# Attribute-call kinds that are model loads regardless of the receiver object.
_ATTR_LOADERS = {"from_pretrained", "load_state_dict"}
_BARE_CALL_NAMES = {"reload_model"}


def _collect_imports(tree: ast.AST) -> Tuple[Dict[str, str], Dict[str, Tuple[str, str]]]:
    """Return (module_aliases, imported_names).

    module_aliases: local name -> canonical module ("t" -> "torch", "torch" -> "torch").
    imported_names: local name -> (canonical module, attr) for `from X import Y [as Z]`
                    (e.g. "load" -> ("torch","load")), or ("__ctor__", ctor) for a model constructor.
    """
    mod_alias: Dict[str, str] = {}
    imported: Dict[str, Tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                canon = a.name.split(".")[0]
                if canon in _MODULE_LOADERS or canon in {"sentence_transformers", "transformers", "paddleocr"}:
                    mod_alias[a.asname or a.name] = canon
        elif isinstance(node, ast.ImportFrom):
            base = (node.module or "").split(".")[0]
            for a in node.names:
                local = a.asname or a.name
                if base in _MODULE_LOADERS and a.name in _MODULE_LOADERS[base]:
                    imported[local] = (base, a.name)          # from torch import load
                elif a.name in _MODEL_CONSTRUCTORS:
                    imported[local] = ("__ctor__", a.name)    # from sentence_transformers import SentenceTransformer
    return mod_alias, imported


class _LoadVisitor(ast.NodeVisitor):
    def __init__(self, mod_alias: Dict[str, str], imported: Dict[str, Tuple[str, str]]) -> None:
        self.mod_alias = mod_alias
        self.imported = imported
        self.stack: List[str] = []
        self.sites: List[Tuple[str, str, int]] = []  # (qualname, kind, lineno)

    def _scoped(self, name: str, node: ast.AST) -> None:
        self.stack.append(name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):  # noqa: D401
        self._scoped(node.name, node)

    def visit_AsyncFunctionDef(self, node):
        self._scoped(node.name, node)

    def visit_ClassDef(self, node):
        self._scoped(node.name, node)

    def visit_Call(self, node: ast.Call) -> None:
        kind = self._classify(node.func)
        if kind:
            qual = ".".join(self.stack) if self.stack else "<module>"
            self.sites.append((qual, kind, node.lineno))
        self.generic_visit(node)

    def _classify(self, func: ast.AST):
        if isinstance(func, ast.Attribute):
            attr = func.attr
            if attr in _ATTR_LOADERS:
                return attr  # from_pretrained / load_state_dict — receiver-independent
            if attr in {"load", "loads"} and isinstance(func.value, ast.Name):
                mod = self.mod_alias.get(func.value.id)
                if mod in _MODULE_LOADERS and attr in _MODULE_LOADERS[mod]:
                    return f"{mod}.{attr}"          # t.load()  (t aliased to torch), p.loads(), etc.
            if attr in _MODEL_CONSTRUCTORS and isinstance(func.value, ast.Name):
                # sentence_transformers.SentenceTransformer(...) via `import sentence_transformers`
                return f"ctor:{attr}"
            return None
        if isinstance(func, ast.Name):
            name = func.id
            if name in self.imported:
                mod, attr = self.imported[name]
                if mod == "__ctor__":
                    return f"ctor:{attr}"           # SentenceTransformer(...) (from-imported)
                return f"{mod}.{attr}"              # bare load() from `from torch import load`
            if name in _MODEL_CONSTRUCTORS:
                return f"ctor:{name}"
            if name in _BARE_CALL_NAMES:
                return f"{name}()"
        return None


def enumerate_sites() -> Dict[str, dict]:
    """{key: {file, symbol, kind, lineno}} for every discovered load site.

    key = "<relpath>::<qualname>::<kind>#<n>" — stable across line-number drift (AST, not grep).
    """
    found: Dict[str, dict] = {}
    for d in SCAN_DIRS:
        for py in sorted((REPO_ROOT / d).rglob("*.py")):
            rel = py.relative_to(REPO_ROOT).as_posix()
            if rel.startswith("tests/") or "/tests/" in f"/{rel}" or py.name.startswith("test_"):
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"), filename=rel)
            except (SyntaxError, UnicodeDecodeError):
                continue
            mod_alias, imported = _collect_imports(tree)
            v = _LoadVisitor(mod_alias, imported)
            v.visit(tree)
            per_key: Dict[str, int] = {}
            for qual, kind, lineno in v.sites:
                base = f"{rel}::{qual}::{kind}"
                n = per_key.get(base, 0)
                per_key[base] = n + 1
                found[f"{base}#{n}"] = {"file": rel, "symbol": qual, "kind": kind, "lineno": lineno}
    return found


def load_manifest() -> Dict[str, dict]:
    if not MANIFEST.exists():
        raise SystemExit(f"[activation-enumerator] manifest missing: {MANIFEST}")
    entries = json.loads(MANIFEST.read_text(encoding="utf-8")).get("sites", {})
    for key, entry in entries.items():
        if entry.get("class") not in VALID_CLASSES:
            raise SystemExit(
                f"[activation-enumerator] manifest entry {key!r} has invalid class "
                f"{entry.get('class')!r} (valid: {sorted(VALID_CLASSES)})"
            )
    return entries


def main(argv=None) -> int:
    found = enumerate_sites()
    manifest = load_manifest()
    unclassified = sorted(set(found) - set(manifest))
    stale = sorted(set(manifest) - set(found))

    if unclassified:
        sys.stderr.write(
            "[activation-enumerator] RED — new UNCLASSIFIED model-load site(s); classify each in "
            "scripts/ci/activation_surface.json before it can land:\n"
        )
        for k in unclassified:
            s = found[k]
            sys.stderr.write(f"  + {s['file']}:{s['lineno']}  {s['kind']}  in {s['symbol']}   key={k}\n")
    if stale:
        sys.stderr.write("[activation-enumerator] RED — STALE manifest entr(y/ies) (re-key/remove):\n")
        for k in stale:
            sys.stderr.write(f"  - {k}  (was: {manifest[k].get('class')})\n")
    if unclassified or stale:
        return 1

    by_class: Dict[str, int] = {}
    for e in manifest.values():
        by_class[e["class"]] = by_class.get(e["class"], 0) + 1
    summary = ", ".join(f"{c}={by_class.get(c, 0)}" for c in sorted(VALID_CLASSES))
    print(f"[activation-enumerator] OK — {len(found)} load sites, all classified ({summary}).")
    gated = sum(1 for e in manifest.values() if e["class"] == "gated")
    print(f"[activation-enumerator] {gated} `gated` production-reachable activation point(s); each "
          "MUST be frozen (Phase A) or route through verify_and_load (Phase B). Until then the "
          "membrane default is #509's unconditional raise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
