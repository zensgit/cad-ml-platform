#!/usr/bin/env python3
"""L3 activation-surface enumerator — bounded completeness for DECLARED loader idioms (not a hand-list).

Discovers each DECLARED-idiom load site that deserializes model bytes into a process (see the
scope note below — this is NOT provably every possible load) and requires each to be
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

Classes (per the L3 design-lock §1): ``gated`` (a conservatively-classified activation site —
per-site logical reachability is audited separately, NOT asserted by this classifier; MUST be
fixed-hash / bundle-digest-checked (Phase A, owner decision (b)) or routed through the proof membrane
(Phase B)), ``producer`` (offline artifact emitter), ``offline`` (a CLI/tool load),
``unmounted`` (0-route scaffold; auto-promotes to gated if mounted), ``infra`` (non-model
deserialization — calibrator / vector-store / cache).

**Scope of the completeness claim (honest — review 6):** this covers the DECLARED loader idioms
listed above (torch/pickle/joblib/onnx `.load(s)` import-alias-aware, `load_state_dict`,
`from_pretrained`, the curated model constructors incl. `InferenceSession`, `reload_model(`). It is
**NOT** a proof of exhaustive coverage of every possible Python model load: a NEW framework or a novel
loader idiom (a bespoke `MyLoader.from_file`, a C-extension entry point, `eval`-constructed calls)
would escape until its pattern is added here. So the guarantee is precisely: *"a new site matching a
DECLARED idiom cannot land unclassified"*, and the idiom list must be widened as new frameworks
appear. Discovery + fail-closed bookkeeping ONLY: it can never emit a "green that enables" — it only
passes (all classified) or reds. Exit 0 = all classified + no stale entry; exit 1 = unclassified/stale
FINDING; exit 2 = MALFUNCTION — the gate could not complete its own check (an in-scope file could not be
read/parsed, or the manifest is missing/unreadable/undecodable/invalid-JSON/schema-invalid), so
completeness CANNOT be asserted → fail-closed, never a silent skip. A malfunction is NOT a finding, and
never wears a finding's exit code. Stdlib only.

**C5 structural wiring check (added in Phase-A):** beyond discovery+classification, each ``gated`` site
carries a ``wiring`` lifecycle (``wired`` / ``gate-before-wired`` / ``latent``) and a raw deserializer
(``torch.load`` / ``pickle.load`` / ``pickle.loads`` / ``joblib.load``) declared ``wired`` must actually
reconstruct from ``activate_file`` / ``activate_bundle`` bytes. The wrap check is a **data-flow binding**,
not function-scope presence (review — F2 false-green): the value the raw loader deserializes must itself
*derive* (via local assignments in the enclosing function) from the bytes returned by
``activate_file`` / ``activate_bundle`` — either the returned value directly, or an in-memory buffer
(``io.BytesIO`` / ``BufferedReader`` …) wrapping it. A function that CALLS the gateway, DISCARDS the
result, and then ``torch.load(path)``\\s straight off a filesystem path is therefore NOT wrapped (its
loaded argument does not derive from the gateway) — this is the F2 discard case, a structural
inconsistency the old presence-only rule scored as a false green. The binding is fail-closed: a local
name counts as gateway-derived only if EVERY assignment to it derives from the gateway (so a
reassignment to a non-gateway value cannot launder it). Per the ratified W4 the check is BLOCKING only
once ``ACTIVATION_ENFORCE_WIRING`` is set (after all in-scope live activations are wired); until the owner
flips it the structural check is present-but-advisory (printed, exit code unchanged) so it cannot red CI
on families that are still ``gate-before-wired``. See :func:`structural_findings`.
"""
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "scripts" / "ci" / "activation_surface.json"
SCAN_DIRS = ("src", "scripts")
VALID_CLASSES = {"gated", "producer", "offline", "unmounted", "infra"}

# --- C5 structural wiring check ---------------------------------------------------------------------
# A raw deserializer (torch.load / pickle.load / pickle.loads / joblib.load) that lands a model into a
# LIVE serving path must not read bytes straight off a path — it must reconstruct from the verified,
# digest-pinned bytes returned by the activation gateway. The gateway functions whose RETURNED VALUE the
# reconstruction must derive from (a data-flow binding, NOT mere function-scope presence — F2):
_ACTIVATION_FUNCS = {"activate_file", "activate_bundle"}
# The one canonical gateway module. A call is a gateway call ONLY if its name RESOLVES to this module
# through the file's import table (bare / aliased / attribute form) — a name-only match let a
# project-local FAKE or shadowed ``def activate_file(...)`` / ``activate_file = lambda ...`` count as
# the real gateway (round-3 false-green). ``_GW_PARENT`` / ``_GW_LEAF`` cover the
# ``from src.core.model_activation import activation_gateway`` module-attribute form.
_CANONICAL_GATEWAY_MODULE = "src.core.model_activation.activation_gateway"
_GW_PARENT, _GW_LEAF = _CANONICAL_GATEWAY_MODULE.rsplit(".", 1)
# In-memory buffer wrappers that may sit between the gateway bytes and the raw loader
# (``torch.load(io.BytesIO(data))``). A raw loader argument that is one of these still counts as
# gateway-derived iff the buffer wraps a gateway-derived value.
_BUFFER_WRAPPERS = {"BytesIO", "BufferedReader", "BufferedRandom"}
# The raw deserializer idioms the wrap check governs (a subset of the enumerated kinds — NOT
# load_state_dict/from_pretrained/ctor:*, which consume bytes/paths the raw loader already produced).
RAW_LOADER_KINDS = {"torch.load", "pickle.load", "pickle.loads", "joblib.load"}
# Every `gated` site carries a wiring lifecycle so the structural check knows its intended posture:
#   wired             — routed through the gateway; its enclosing function MUST delegate to activate_*.
#   gate-before-wired — a LIVE gated site not yet routed through the gateway (deferred to a later wave,
#                       e.g. the OCR/embedding families that need C1 bundle-path support); still a raw
#                       load today, tracked so it cannot be forgotten.
#   latent            — a sealed / not-currently-live path (e.g. the 403-sealed hot-reload); raw, exempt.
VALID_WIRING = {"wired", "gate-before-wired", "latent"}
# W4 (ratified): the structural wrap check is BLOCKING only after all in-scope live activations are
# wired. Until the owner flips it, it is present-but-advisory; setting this env truthy makes a
# structural inconsistency a FINDING (exit 1). See main().
ENV_ENFORCE_WIRING = "ACTIVATION_ENFORCE_WIRING"

# Deserializer / model-loader modules and their load attributes.
_MODULE_LOADERS = {"torch": {"load"}, "pickle": {"load", "loads"}, "joblib": {"load"}, "onnx": {"load"}}
# Model constructors that load weights on construction (import-aware; curated — extend as needed).
_MODEL_CONSTRUCTORS = {"SentenceTransformer", "CrossEncoder", "PaddleOCR", "InferenceSession"}
# Extra modules whose aliases we track (for module.Constructor(...) forms like ort.InferenceSession).
_TRACKED_EXTRA_MODULES = {"sentence_transformers", "transformers", "paddleocr", "onnxruntime"}
# Attribute-call kinds that are model loads regardless of the receiver object.
_ATTR_LOADERS = {"from_pretrained", "load_state_dict"}
_BARE_CALL_NAMES = {"reload_model"}

# Exit codes — a gate MALFUNCTION must never be mistaken for a clean pass (0) or a finding (1).
EXIT_OK = 0            # every discovered site is classified, no stale manifest entry
EXIT_FINDING = 1       # a new UNCLASSIFIED site, or a STALE manifest entry
EXIT_MALFUNCTION = 2   # the gate could not complete its own check — an unreadable/unparseable
                       # in-scope file, OR an unusable manifest → completeness UNASSERTABLE →
                       # fail-closed (never a silent skip, never an "empty but valid" degrade)


class EnumeratorMalfunction(Exception):
    """The gate could not complete its own check, so completeness CANNOT be asserted.

    Two sources, one verdict — a MALFUNCTION (exit 2), never a finding (exit 1):

    * **source scan** — an in-scope file could not be read or parsed; it may hide unregistered
      loaders, so skipping it would be fail-open;
    * **manifest** — missing / unreadable / undecodable / invalid JSON / schema-invalid /
      invalid class; without a usable classification there is nothing to check completeness
      against, and degrading to an "empty but valid" manifest would fabricate a clean pass.

    Fail closed in both cases.
    """

    def __init__(self, malfunctions: List[Tuple[str, str, str]]) -> None:
        self.malfunctions = malfunctions  # list of (relpath, error_type, message)
        super().__init__(f"{len(malfunctions)} gate malfunction(s)")


def _collect_imports(
    tree: ast.AST,
) -> Tuple[Dict[str, str], Dict[str, Tuple[str, str]], set, set]:
    """Return (module_aliases, imported_names, gw_names, gw_modules).

    module_aliases: local name -> canonical module ("t" -> "torch", "torch" -> "torch").
    imported_names: local name -> (canonical module, attr) for `from X import Y [as Z]`
                    (e.g. "load" -> ("torch","load")), or ("__ctor__", ctor) for a model constructor.
    gw_names:   local names bound to a canonical gateway FUNCTION — for a Name-form call
                ``activate_file(...)`` — from ``from src.core.model_activation.activation_gateway import
                activate_file [as af]`` (the import may be module- OR function-scoped; ``ast.walk`` sees
                both).
    gw_modules: local names bound to the canonical gateway MODULE — for an Attribute-form call
                ``gw.activate_file(...)`` — from ``import ...activation_gateway as gw`` or
                ``from src.core.model_activation import activation_gateway [as gw]``.
    NB: gateway names are tracked SEPARATELY from ``imported_names`` on purpose — folding them in would
    make ``_classify`` misread a gateway call as a loader kind.
    """
    mod_alias: Dict[str, str] = {}
    imported: Dict[str, Tuple[str, str]] = {}
    gw_names: set = set()
    gw_modules: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                canon = a.name.split(".")[0]
                if canon in _MODULE_LOADERS or canon in _TRACKED_EXTRA_MODULES:
                    mod_alias[a.asname or a.name] = canon
                if a.name == _CANONICAL_GATEWAY_MODULE and a.asname:
                    # import src.core.model_activation.activation_gateway as gw  -> gw is the module
                    gw_modules.add(a.asname)
        elif isinstance(node, ast.ImportFrom):
            full = node.module or ""
            base = full.split(".")[0]
            for a in node.names:
                local = a.asname or a.name
                if base in _MODULE_LOADERS and a.name in _MODULE_LOADERS[base]:
                    imported[local] = (base, a.name)          # from torch import load
                elif a.name in _MODEL_CONSTRUCTORS:
                    imported[local] = ("__ctor__", a.name)    # from sentence_transformers import SentenceTransformer
                if full == _CANONICAL_GATEWAY_MODULE and a.name in _ACTIVATION_FUNCS:
                    gw_names.add(local)                       # from ...activation_gateway import activate_file
                elif full == _GW_PARENT and a.name == _GW_LEAF:
                    gw_modules.add(local)                     # from src.core.model_activation import activation_gateway
    return mod_alias, imported, gw_names, gw_modules


# A block path is a tuple of ``(id(compound_node), field)`` pairs from the enclosing function body
# (root = ``()``) down to the block a statement lives in. Using the AST node's identity (NOT an
# incrementing counter) makes paths computed by two independent traversals COMPARABLE by prefix, and
# keeps sibling blocks distinct — an ``if``'s ``body`` and ``orelse`` never share a prefix, so a binding
# in one branch cannot dominate a load in the other. Statement-list fields that open a new block:
_BLOCK_FIELDS = ("body", "orelse", "finalbody")


class _GatewayAssignScan(ast.NodeVisitor):
    """Collect ``name = <expr>`` bindings in ONE function body, NOT descending into nested function
    scopes (a nested function's locals are not the enclosing function's). Used to resolve which local
    names carry gateway-derived bytes. Handles plain assign, annotated assign, and walrus.

    Records each binding's ``(lineno, block_path)`` and a ``block_of`` map ``id(node) -> block_path`` for
    EVERY visited node so the load site's block can be looked up later — this is what makes the wrap
    decision DOMINANCE/lexical-order aware (F2 round-2): a gateway binding counts for a load site only if
    it lexically precedes AND its block dominates (is the same block or an ancestor block of) the load."""

    def __init__(self) -> None:
        # (target name, RHS value node, lineno, block_path), in source order:
        self.assigns: List[Tuple[str, ast.AST, int, tuple]] = []
        # id(node) -> block_path, for every node reached in this scope (load sites are looked up here):
        self.block_of: Dict[int, tuple] = {}
        self._path: tuple = ()

    def visit(self, node):  # tag every node with the block active when it is reached, then dispatch.
        self.block_of[id(node)] = self._path
        return super().visit(node)

    def visit_FunctionDef(self, node):  # a nested def — its body is a different scope; do not descend.
        return

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        return

    def visit_Assign(self, node):
        for t in node.targets:
            if isinstance(t, ast.Name):
                self.assigns.append((t.id, node.value, node.lineno, self._path))
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name) and node.value is not None:
            self.assigns.append((node.target.id, node.value, node.lineno, self._path))
        self.generic_visit(node)

    def visit_NamedExpr(self, node):  # (data := activate_file(...))
        if isinstance(node.target, ast.Name):
            self.assigns.append((node.target.id, node.value, node.lineno, self._path))
        self.generic_visit(node)

    def _visit_compound(self, node):
        """A statement that opens sub-blocks (if/for/while/with/try/class/match). Visit its header
        (non-block) parts at the CURRENT block, then recurse into each sub-statement-list with a fresh,
        DISTINCT block path keyed by ``(id(node), field)`` so branches never share a dominating prefix."""
        for field, value in ast.iter_fields(node):
            if field in _BLOCK_FIELDS:
                continue
            if field == "handlers":  # try/except: each handler body is its own block
                for h in value or []:
                    self._descend(h.body, (id(h), "body"))
                continue
            if field == "cases":  # match/case (3.10+): each case body is its own block
                for c in value or []:
                    self._descend(c.body, (id(c), "body"))
                continue
            for child in (value if isinstance(value, list) else [value]):
                if isinstance(child, ast.AST):
                    self.visit(child)
        for field in _BLOCK_FIELDS:
            body = getattr(node, field, None)
            if isinstance(body, list):
                self._descend(body, (id(node), field))

    def _descend(self, stmts, key) -> None:
        saved = self._path
        self._path = saved + (key,)
        for s in stmts:
            self.visit(s)
        self._path = saved

    visit_If = _visit_compound
    visit_For = _visit_compound
    visit_AsyncFor = _visit_compound
    visit_While = _visit_compound
    visit_With = _visit_compound
    visit_AsyncWith = _visit_compound
    visit_Try = _visit_compound
    visit_TryStar = _visit_compound          # 3.11+ (harmless if ast has no TryStar)
    visit_ClassDef = _visit_compound         # descend class body as a block; its methods are skipped
    visit_Match = _visit_compound            # 3.10+


class _LoadVisitor(ast.NodeVisitor):
    def __init__(
        self,
        mod_alias: Dict[str, str],
        imported: Dict[str, Tuple[str, str]],
        gw_names: set,
        gw_modules: set,
    ) -> None:
        self.mod_alias = mod_alias
        self.imported = imported
        # Local names that resolve to the canonical gateway (function-form / module-form) — a call is a
        # gateway call ONLY via these, never by bare spelling (round-3: a local fake activate_file).
        self.gw_names = gw_names
        self.gw_modules = gw_modules
        self.stack: List[str] = []
        # full-qualname of every enclosing FUNCTION scope (class scopes excluded) — used to attribute
        # each load site to its enclosing function(s).
        self.func_qual: List[str] = []
        # (qualname, kind, lineno, wrapped) — wrapped = the loaded value derives (data-flow) from the
        # activation gateway (only meaningful, and only computed, for RAW_LOADER_KINDS).
        self.sites: List[Tuple[str, str, int, bool]] = []
        # Per enclosing function: name -> list of (lineno, block_path) for each gateway-derived binding
        # of that name, bound WITHIN THAT function only. Outer-scope bindings are deliberately NOT
        # inherited into nested scopes (round-3 false-green fix): absent real cross-scope call-order
        # analysis, an outer binding cannot be proven to dominate a nested loader. Top of stack = current
        # scope.
        self.func_gwvars: List[Dict[str, List[Tuple[int, tuple]]]] = []
        # Per enclosing function: id(node) -> block_path, so a load site's block can be looked up to
        # decide dominance. Parallel to func_gwvars.
        self.func_blockof: List[Dict[int, tuple]] = []

    def _scoped(self, name: str, node: ast.AST, is_func: bool) -> None:
        self.stack.append(name)
        if is_func:
            self.func_qual.append(".".join(self.stack))
            gwvars, block_of = self._collect_gateway_vars(node)
            # Do NOT inherit outer-scope gateway vars into this nested function scope (round-3 false-green
            # fix). The old code relaxed every parent binding to (0, ()) so it ALWAYS dominated — which let
            # a nested closure that does ``torch.load(x)`` FIRST (before, or regardless of, an outer
            # ``x = activate_file(...)`` binding) score wrapped=True, though the enumerator cannot prove the
            # outer binding runs before the nested load. Absent real cross-scope call-order analysis, the
            # conservative fail-closed rule is: a raw loader inside a nested function/lambda is wrapped ONLY
            # if a gateway binding WITHIN THAT SAME function dominates it. So each scope carries only its
            # OWN bindings.
            self.func_gwvars.append(gwvars)
            self.func_blockof.append(block_of)
        self.generic_visit(node)
        if is_func:
            self.func_gwvars.pop()
            self.func_blockof.pop()
            self.func_qual.pop()
        self.stack.pop()

    def visit_FunctionDef(self, node):  # noqa: D401
        self._scoped(node.name, node, True)

    def visit_AsyncFunctionDef(self, node):
        self._scoped(node.name, node, True)

    def visit_ClassDef(self, node):
        self._scoped(node.name, node, False)

    def visit_Call(self, node: ast.Call) -> None:
        kind = self._classify(node.func)
        if kind:
            qual = ".".join(self.stack) if self.stack else "<module>"
            # `wrapped`: does the deserialized value derive (reaching-definition, within the enclosing
            # function) from activate_file / activate_bundle bytes? Only raw deserializers take such a
            # bytes/stream argument; for the receiver-independent kinds (load_state_dict / from_pretrained
            # / ctor:*) the wrap requirement does not apply and `wrapped` is left False (unused downstream).
            wrapped = False
            if kind in RAW_LOADER_KINDS and node.args:
                gwvars = self.func_gwvars[-1] if self.func_gwvars else {}
                block_of = self.func_blockof[-1] if self.func_blockof else {}
                site_block = block_of.get(id(node), ())
                wrapped = self._derives_from_gateway(node.args[0], gwvars, node.lineno, site_block)
            self.sites.append((qual, kind, node.lineno, wrapped))
        self.generic_visit(node)

    def _collect_gateway_vars(
        self, func_node: ast.AST
    ) -> Tuple[Dict[str, List[Tuple[int, tuple]]], Dict[int, tuple]]:
        """(gwvars, block_of) for ``func_node``'s body.

        ``gwvars`` maps each local name that carries gateway-derived bytes to the list of
        ``(lineno, block_path)`` of its bindings; ``block_of`` maps ``id(node) -> block_path`` for load-site
        lookup.

        Membership is position-INSENSITIVE and fail-closed under reassignment: a name counts ONLY if it is
        bound at least once and EVERY one of its bindings derives from the gateway — so
        ``data = activate_file(...); data = open(p).read()`` does NOT launder ``data`` into a gateway var.
        Fixpoint so buffer/name chains resolve (``data = activate_file(...); buf = io.BytesIO(data)`` ->
        both gateway-derived). Whether a binding actually counts FOR A GIVEN load site is decided later by
        dominance (:meth:`_derives_from_gateway`) using the recorded ``(lineno, block_path)`` — a binding
        that lexically follows or does not dominate the load does not count (F2 round-2 ordering fix)."""
        scan = _GatewayAssignScan()
        for stmt in getattr(func_node, "body", []):
            scan.visit(stmt)
        by_name: Dict[str, List[Tuple[ast.AST, int, tuple]]] = {}
        for name, val, lineno, block in scan.assigns:
            by_name.setdefault(name, []).append((val, lineno, block))
        gwnames: set = set()
        changed = True
        while changed:
            changed = False
            for name, binds in by_name.items():
                if name in gwnames:
                    continue
                if binds and all(self._expr_is_gateway(v, gwnames) for (v, _, _) in binds):
                    gwnames.add(name)
                    changed = True
        # Every binding of a gwname is gateway-derived (all-bindings rule), so all its positions admit;
        # dominance at the load site picks the ones that actually reach it.
        gwvars = {name: [(ln, blk) for (_, ln, blk) in by_name[name]] for name in gwnames}
        return gwvars, scan.block_of

    def _expr_is_gateway(self, expr: ast.AST, gwnames: set) -> bool:
        """Position-insensitive membership check used by the fixpoint: does ``expr`` derive from the
        gateway given the set of names already proven gateway-derived? (No dominance here — this only
        establishes which NAMES can ever be gateway-derived; dominance is applied per load site.)"""
        if isinstance(expr, ast.Name):
            return expr.id in gwnames
        if isinstance(expr, ast.Call):
            if self._is_activation_call(expr.func):
                return True
            if self._is_buffer_wrapper(expr.func):
                return any(self._expr_is_gateway(a, gwnames) for a in expr.args)
        return False

    def _derives_from_gateway(
        self, expr: ast.AST, gwvars: Dict[str, List[Tuple[int, tuple]]], site_lineno: int, site_block: tuple
    ) -> bool:
        """True iff ``expr`` is (transitively) the gateway bytes AT this load site — i.e. reconstructed
        from a gateway binding that DOMINATES the site. Three cases:

        * an inline gateway call (``torch.load(activate_file(...))``) — always counts, regardless of line;
        * an in-memory buffer wrapping such a value (``io.BytesIO(...)``) — recurse into its args;
        * a local name proven gateway-derived AND bound by at least one binding that dominates the site.

        A name whose only gateway binding LEXICALLY FOLLOWS the load (F2 round-2: the gateway is called
        AFTER the loader) does NOT count — the reaching value at the loader is not the gateway bytes. A
        filesystem path/str/Path/attribute (e.g. ``self.model_path``) never counts. Fail-closed: an
        ``ANY``-dominating binding suffices because membership already guarantees EVERY binding of the name
        is gateway-derived, so any binding that dominates proves the reaching definition is gateway-verified
        (requiring ALL would false-RED legitimately-wrapped multi-binding sites; on the real tree every
        gwvar is bound exactly once, so ANY and ALL coincide)."""
        if isinstance(expr, ast.Name):
            return any(self._dominates(bl, bp, site_lineno, site_block) for bl, bp in gwvars.get(expr.id, ()))
        if isinstance(expr, ast.Call):
            if self._is_activation_call(expr.func):
                return True
            if self._is_buffer_wrapper(expr.func):
                return any(
                    self._derives_from_gateway(a, gwvars, site_lineno, site_block) for a in expr.args
                )
        return False

    @staticmethod
    def _dominates(bind_lineno: int, bind_block: tuple, site_lineno: int, site_block: tuple) -> bool:
        """A gateway binding counts for a load site only if it lexically PRECEDES the site AND its block
        DOMINATES the site's block (same block or an ancestor — ``bind_block`` is a prefix of
        ``site_block``). A binding at/after the load line, or in a sibling branch/loop that does not enclose
        the load, does NOT count. Conservative / fail-closed."""
        if bind_lineno >= site_lineno:
            return False
        return site_block[: len(bind_block)] == bind_block

    @staticmethod
    def _is_buffer_wrapper(func: ast.AST) -> bool:
        if isinstance(func, ast.Attribute):
            return func.attr in _BUFFER_WRAPPERS  # io.BytesIO(...)
        if isinstance(func, ast.Name):
            return func.id in _BUFFER_WRAPPERS      # BytesIO(...) (from io import BytesIO)
        return False

    def _is_activation_call(self, func: ast.AST) -> bool:
        """A gateway call ONLY if the name RESOLVES (via this file's import table) to the canonical
        gateway module — NOT by bare spelling. ``activate_file(...)`` counts iff ``activate_file`` was
        imported from ``src.core.model_activation.activation_gateway`` (bare/aliased); an attribute call
        ``gw.activate_file(...)`` counts iff ``gw`` is a canonical-gateway-module alias. A project-local
        or differently-sourced ``activate_file`` (a fake, a shadowing ``def``/``lambda``) does NOT."""
        if isinstance(func, ast.Name):
            return func.id in self.gw_names
        if isinstance(func, ast.Attribute):
            return (
                func.attr in _ACTIVATION_FUNCS
                and isinstance(func.value, ast.Name)
                and func.value.id in self.gw_modules
            )
        return False

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
    malfunctions: List[Tuple[str, str, str]] = []
    for d in SCAN_DIRS:
        for py in sorted((REPO_ROOT / d).rglob("*.py")):
            rel = py.relative_to(REPO_ROOT).as_posix()
            if rel.startswith("tests/") or "/tests/" in f"/{rel}" or py.name.startswith("test_"):
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"), filename=rel)
            except (SyntaxError, ValueError, OSError) as exc:
                # FAIL-CLOSED: an unparseable OR unreadable in-scope file means we cannot prove we saw
                # every loader in it. ValueError covers UnicodeDecodeError and Python versions where
                # ast.parse reports a null character that way. Record a MALFUNCTION (exit 2) — never
                # silently skip (fail-open).
                malfunctions.append((rel, type(exc).__name__, " ".join(str(exc).split())))
                continue
            mod_alias, imported, gw_names, gw_modules = _collect_imports(tree)
            v = _LoadVisitor(mod_alias, imported, gw_names, gw_modules)
            v.visit(tree)
            per_key: Dict[str, int] = {}
            for qual, kind, lineno, wrapped in v.sites:
                base = f"{rel}::{qual}::{kind}"
                n = per_key.get(base, 0)
                per_key[base] = n + 1
                # `wrapped` (computed by the visitor as a data-flow binding, NOT function-scope presence):
                # the raw loader's deserialized argument derives from activate_file / activate_bundle bytes.
                found[f"{base}#{n}"] = {
                    "file": rel, "symbol": qual, "kind": kind, "lineno": lineno, "wrapped": wrapped,
                }
    if malfunctions:
        raise EnumeratorMalfunction(malfunctions)
    return found


def load_manifest() -> Dict[str, dict]:
    """The classification manifest.

    EVERY failure here is a gate MALFUNCTION (exit 2), never a finding (exit 1): if the manifest is
    missing / unreadable / undecodable / not valid JSON / schema-invalid, the gate cannot assert
    completeness at all. Previously these raised ``SystemExit(str)`` and surfaced as exit 1 — i.e. a
    malfunction wearing a finding's exit code, which breaks the exit-code contract.
    """
    rel = MANIFEST.as_posix()
    if not MANIFEST.exists():
        raise EnumeratorMalfunction(
            [(rel, "FileNotFoundError", "manifest missing — completeness cannot be asserted")]
        )
    try:
        raw = MANIFEST.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EnumeratorMalfunction([(rel, type(exc).__name__, " ".join(str(exc).split()))]) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EnumeratorMalfunction([(rel, "JSONDecodeError", " ".join(str(exc).split()))]) from exc
    if (
        not isinstance(data, dict)
        or "sites" not in data
        or not isinstance(data["sites"], dict)
    ):
        # NB: `data.get("sites", {})` would let a manifest with NO 'sites' key through as an "empty
        # but valid" manifest — a fabricated clean pass. A missing key is schema-invalid, not empty.
        raise EnumeratorMalfunction(
            [(rel, "SchemaError",
              "manifest must be an object with a 'sites' object (a missing 'sites' key is "
              "schema-invalid, NOT an empty manifest)")]
        )
    entries = data["sites"]
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            raise EnumeratorMalfunction(
                [(rel, "SchemaError", f"entry {key!r} must be an object, got {type(entry).__name__}")]
            )
        if entry.get("class") not in VALID_CLASSES:
            raise EnumeratorMalfunction(
                [(rel, "InvalidClass",
                  f"entry {key!r} has invalid class {entry.get('class')!r} "
                  f"(valid: {sorted(VALID_CLASSES)})")]
            )
    return entries


def structural_findings(
    found: Dict[str, dict], manifest: Dict[str, dict]
) -> List[Tuple[str, str, str]]:
    """Wiring-consistency findings for `gated` sites — the C5 structural proof.

    Returns ``(key, code, message)`` for each inconsistency between a gated site's declared ``wiring``
    lifecycle and what the source AST actually shows:

    * a gated site with no valid ``wiring`` value — the lifecycle must be declared, else a new gated
      load could hide unrouted;
    * a raw deserializer (``RAW_LOADER_KINDS``) marked ``wiring="wired"`` whose deserialized argument does
      NOT derive (data-flow) from ``activate_file``/``activate_bundle`` bytes — i.e. it can still read
      bytes straight off a path, so a missing/tampered artifact could load UNVERIFIED. This catches BOTH
      the remove-the-wrapper regression (no gateway call at all) AND the F2 discard case (the function
      calls the gateway, ignores the return, then ``torch.load(path)``);
    * a raw deserializer marked ``gate-before-wired``/``latent`` that IS wrapped — the marker is stale, it
      has been routed through the gateway and should be reclassified ``wired``.

    Sites of class other than ``gated`` (producer / offline / unmounted / infra) are not serving-path
    activations and are out of scope here. Non-raw gated kinds (``load_state_dict`` / ``from_pretrained`` /
    ``ctor:*``) still require a ``wiring`` value but are not subject to the wrap requirement — they consume
    what a raw loader already produced. Whether these findings are BLOCKING is decided by the caller
    (per the ratified W4): advisory until ``ENV_ENFORCE_WIRING`` is set.
    """
    findings: List[Tuple[str, str, str]] = []
    for key, site in found.items():
        entry = manifest.get(key)
        if entry is None or entry.get("class") != "gated":
            continue  # unclassified sites are caught by the completeness check; non-gated is out of scope
        wiring = entry.get("wiring")
        if wiring not in VALID_WIRING:
            findings.append(
                (key, "missing-wiring",
                 f"gated site needs a `wiring` in {sorted(VALID_WIRING)}, got {wiring!r}")
            )
            continue
        if site["kind"] not in RAW_LOADER_KINDS:
            continue  # the wrap requirement governs raw deserializers only
        wrapped = bool(site.get("wrapped"))
        if wiring == "wired" and not wrapped:
            findings.append(
                (key, "wired-but-unwrapped",
                 "marked wiring=wired but its deserialized argument does not derive (data-flow) from "
                 "activate_file/activate_bundle bytes — a raw load-by-path can load UNVERIFIED bytes "
                 "(the gateway may be called then its result discarded — the F2 false-green)")
            )
        elif wiring in {"gate-before-wired", "latent"} and wrapped:
            findings.append(
                (key, "unwired-but-wrapped",
                 f"marked wiring={wiring} but the site IS routed through the activation gateway — "
                 "reclassify to wiring=wired")
            )
    return findings


def _enforce_wiring_enabled() -> bool:
    """Whether the structural wrap check is BLOCKING (per ratified W4). Default: advisory-only."""
    return os.environ.get(ENV_ENFORCE_WIRING, "").strip().lower() in {"1", "true", "yes", "on"}


def main(argv=None) -> int:
    try:
        found = enumerate_sites()
        manifest = load_manifest()
    except EnumeratorMalfunction as exc:
        sys.stderr.write(
            "[activation-enumerator] MALFUNCTION (exit 2) — the gate could not complete its own check "
            "(an unreadable/unparseable in-scope file, or an unusable manifest), so completeness CANNOT "
            "be asserted. This is a gate malfunction, NOT a finding (exit 1):\n"
        )
        for rel, etype, msg in exc.malfunctions:
            sys.stderr.write(f"  ! {rel}  {etype}: {msg}\n")
        return EXIT_MALFUNCTION
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

    # C5 structural wiring check. Per the ratified W4 it is BLOCKING only once ENV_ENFORCE_WIRING is
    # set (after all in-scope live activations are wired); until then it is present-but-advisory so it
    # cannot red CI on account of families that are still gate-before-wired.
    struct = structural_findings(found, manifest)
    enforce = _enforce_wiring_enabled()
    if struct:
        if enforce:
            sys.stderr.write(
                "[activation-enumerator] RED (structural wiring, ENFORCED) — gated raw-loader routing "
                "inconsistenc(y/ies); a LIVE raw loader must reconstruct from activate_file/activate_bundle "
                "bytes:\n"
            )
        else:
            sys.stderr.write(
                "[activation-enumerator] ADVISORY (structural wiring, non-blocking — set "
                f"{ENV_ENFORCE_WIRING}=1 to enforce per ratified W4) — gated raw-loader routing "
                "inconsistenc(y/ies):\n"
            )
        for k, code, msg in struct:
            sys.stderr.write(f"  * {k}  [{code}]  {msg}\n")

    if unclassified or stale or (struct and enforce):
        return EXIT_FINDING

    by_class: Dict[str, int] = {}
    for e in manifest.values():
        by_class[e["class"]] = by_class.get(e["class"], 0) + 1
    summary = ", ".join(f"{c}={by_class.get(c, 0)}" for c in sorted(VALID_CLASSES))
    print(f"[activation-enumerator] OK — {len(found)} load sites, all classified ({summary}).")
    gated = sum(1 for e in manifest.values() if e["class"] == "gated")
    print(f"[activation-enumerator] {gated} `gated` AST load site(s) — a CONSERVATIVE classification "
          "that may include latent / not-yet-proven-reachable sites; per-site logical reachability is "
          "audited separately, NOT asserted here. These are DISCOVERED + classified only; Phase A0 does "
          "NOT pin them (they still load). Under full Phase A each MUST be fixed-hash- or "
          "bundle-digest-checked (owner decision (b)) or routed through verify_and_load (Phase B).")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
