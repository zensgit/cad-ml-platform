"""The L3 activation-surface enumerator is fail-closed: a NEW un-annotated model-load site reds CI.

This is the "completeness by construction, not by a hand-list" guarantee of the L3 design-lock — a
new activation surface cannot land silently; it must be classified in activation_surface.json first.
"""
from __future__ import annotations

import json
import importlib

import pytest

enum = importlib.import_module("scripts.ci.activation_surface_enumerator")


def test_current_tree_is_fully_classified() -> None:
    # every discovered load site has a manifest entry and every manifest entry still resolves.
    found = set(enum.enumerate_sites())
    manifest = set(enum.load_manifest())
    assert found == manifest, (
        f"unclassified (new): {sorted(found - manifest)}\nstale (manifest-only): {sorted(manifest - found)}"
    )
    assert enum.main() == 0


def test_new_unclassified_load_site_reds(tmp_path, monkeypatch, capsys) -> None:
    # simulate a NEW load site by pointing the enumerator at a manifest that is MISSING one entry
    # that the real tree has -> the site is "unclassified" -> exit 1.
    real = json.loads(enum.MANIFEST.read_text(encoding="utf-8"))
    dropped_key = sorted(real["sites"])[0]
    real["sites"].pop(dropped_key)
    partial = tmp_path / "activation_surface.json"
    partial.write_text(json.dumps(real), encoding="utf-8")
    monkeypatch.setattr(enum, "MANIFEST", partial)
    assert enum.main() == 1
    err = capsys.readouterr().err
    assert "UNCLASSIFIED" in err and dropped_key in err


def test_stale_manifest_entry_reds(tmp_path, monkeypatch, capsys) -> None:
    # a manifest entry that resolves to no load site (e.g. a deleted/renamed loader) reds too.
    real = json.loads(enum.MANIFEST.read_text(encoding="utf-8"))
    real["sites"]["src/ml/ghost.py::Ghost.load::torch.load#0"] = {"class": "gated", "reason": "phantom"}
    stale_manifest = tmp_path / "activation_surface.json"
    stale_manifest.write_text(json.dumps(real), encoding="utf-8")
    monkeypatch.setattr(enum, "MANIFEST", stale_manifest)
    assert enum.main() == 1
    assert "STALE" in capsys.readouterr().err


def test_invalid_class_is_rejected(tmp_path, monkeypatch, capsys) -> None:
    real = json.loads(enum.MANIFEST.read_text(encoding="utf-8"))
    k = sorted(real["sites"])[0]
    real["sites"][k] = {"class": "totally-safe-trust-me", "reason": "x"}
    bad = tmp_path / "activation_surface.json"
    bad.write_text(json.dumps(real), encoding="utf-8")
    monkeypatch.setattr(enum, "MANIFEST", bad)
    # a schema-invalid manifest is a MALFUNCTION (exit 2), not a finding (exit 1): previously this
    # raised SystemExit(str) and the CLI surfaced exit 1 — a malfunction wearing a finding's code.
    with pytest.raises(enum.EnumeratorMalfunction):
        enum.load_manifest()
    assert enum.main() == enum.EXIT_MALFUNCTION == 2
    assert "InvalidClass" in capsys.readouterr().err


def test_every_gated_site_carries_a_family() -> None:
    # a gated (conservatively-classified) site must name its model family, so the membrane can bind it.
    manifest = enum.load_manifest()
    missing = [k for k, e in manifest.items() if e["class"] == "gated" and not e.get("family")]
    assert not missing, f"gated sites without a family: {missing}"


# --- fail-closed on unparseable files: a MALFUNCTION (exit 2), never a silent skip -----------------
# observed-RED: on the pre-fix code these unparseable files were silently `continue`d, so with an empty
# manifest `main()` returned 0 (clean) — an unparseable file could hide an unregistered loader. The fix
# raises EnumeratorMalfunction -> exit 2 (a malfunction, distinct from the exit-1 finding path).

def _point_scan_at(tmp_path, monkeypatch, files: dict) -> None:
    """Point the enumerator's scan at a temp tree. files = {relpath: str|bytes}. Empty manifest, so on
    the OLD (skip) behaviour main() would have returned 0 — making the exit-2 assertion an observed-RED."""
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content) if isinstance(content, bytes) else p.write_text(content, encoding="utf-8")
    mani = tmp_path / "manifest.json"
    mani.write_text(json.dumps({"sites": {}}), encoding="utf-8")
    monkeypatch.setattr(enum, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(enum, "SCAN_DIRS", ("src",))
    monkeypatch.setattr(enum, "MANIFEST", mani)


def test_syntaxerror_in_scope_is_malfunction_exit_2(tmp_path, monkeypatch, capsys) -> None:
    _point_scan_at(tmp_path, monkeypatch, {"src/broken_syntax.py": "def f(:\n    pass\n"})
    assert enum.main() == enum.EXIT_MALFUNCTION == 2
    err = capsys.readouterr().err
    assert "MALFUNCTION" in err and "src/broken_syntax.py" in err and "SyntaxError" in err
    assert "NOT a finding" in err  # must not be confused with the exit-1 finding path


def test_unicodedecodeerror_in_scope_is_malfunction_exit_2(tmp_path, monkeypatch, capsys) -> None:
    _point_scan_at(tmp_path, monkeypatch, {"src/bad_bytes.py": b"\xff\xfe not valid utf-8 \x80\x81\n"})
    assert enum.main() == enum.EXIT_MALFUNCTION == 2
    err = capsys.readouterr().err
    assert "MALFUNCTION" in err and "src/bad_bytes.py" in err and "UnicodeDecodeError" in err


def test_valueerror_from_ast_parse_is_malfunction_exit_2(tmp_path, monkeypatch, capsys) -> None:
    _point_scan_at(tmp_path, monkeypatch, {"src/null_byte.py": "X = 1\n"})

    def reject_null_byte(*_args, **_kwargs) -> None:
        raise ValueError("source code string cannot contain null bytes")

    monkeypatch.setattr(enum.ast, "parse", reject_null_byte)
    assert enum.main() == enum.EXIT_MALFUNCTION == 2
    err = capsys.readouterr().err
    assert "MALFUNCTION" in err and "src/null_byte.py" in err and "ValueError" in err


def test_unreadable_in_scope_file_is_malfunction_exit_2(tmp_path, monkeypatch, capsys) -> None:
    # an OSError while reading an in-scope file is equally "cannot prove I saw its loaders".
    # A directory named *.py is matched by rglob and raises IsADirectoryError (an OSError) on read —
    # deterministic and portable (unlike chmod, which root ignores).
    _point_scan_at(tmp_path, monkeypatch, {"src/fine.py": "X = 1\n"})
    (tmp_path / "src" / "unreadable.py").mkdir()
    assert enum.main() == enum.EXIT_MALFUNCTION == 2
    err = capsys.readouterr().err
    assert "MALFUNCTION" in err and "src/unreadable.py" in err


def test_wellformed_file_is_not_a_malfunction(tmp_path, monkeypatch) -> None:
    # positive control: a parseable in-scope file with no loader is NOT a malfunction -> exit 0, not 2.
    _point_scan_at(tmp_path, monkeypatch, {"src/fine.py": "X = 1\ndef g():\n    return X\n"})
    assert enum.enumerate_sites() == {}          # parses cleanly, no raise, no loaders
    assert enum.main() == enum.EXIT_OK == 0


# --- manifest failures are MALFUNCTIONS too (exit 2), never findings (exit 1) ----------------------
# observed-RED: on the pre-fix code all three below raised SystemExit(str)/JSONDecodeError and the CLI
# exited 1 — i.e. a gate malfunction misclassified as a finding. Reproduced: exit 1, 1, 1.

def test_missing_manifest_is_malfunction_exit_2(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(enum, "MANIFEST", tmp_path / "does_not_exist.json")
    assert enum.main() == enum.EXIT_MALFUNCTION == 2
    err = capsys.readouterr().err
    assert "MALFUNCTION" in err and "FileNotFoundError" in err and "NOT a finding" in err


def test_corrupt_json_manifest_is_malfunction_exit_2(tmp_path, monkeypatch, capsys) -> None:
    bad = tmp_path / "activation_surface.json"
    bad.write_text("{ this is not valid json", encoding="utf-8")
    monkeypatch.setattr(enum, "MANIFEST", bad)
    assert enum.main() == enum.EXIT_MALFUNCTION == 2
    assert "JSONDecodeError" in capsys.readouterr().err


def test_manifest_schema_violation_is_malfunction_exit_2(tmp_path, monkeypatch, capsys) -> None:
    bad = tmp_path / "activation_surface.json"
    bad.write_text(json.dumps({"sites": {"k::x::torch.load#0": "not-an-object"}}), encoding="utf-8")
    monkeypatch.setattr(enum, "MANIFEST", bad)
    assert enum.main() == enum.EXIT_MALFUNCTION == 2
    assert "SchemaError" in capsys.readouterr().err


@pytest.mark.parametrize("payload", ["{}", '{"notsites": {}}', '{"sites": []}', '"a string"'])
def test_manifest_without_a_sites_object_is_malfunction_exit_2(
    payload, tmp_path, monkeypatch, capsys
) -> None:
    # observed-RED: `data.get("sites", {})` let a manifest with NO 'sites' key through as an "empty
    # but valid" manifest — in a scan scope with no loaders that printed "OK ... all classified" and
    # exited 0, i.e. a schema-invalid manifest fabricating a clean pass. A missing/!dict 'sites' is a
    # MALFUNCTION (exit 2), never an empty manifest.
    bad = tmp_path / "activation_surface.json"
    bad.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(enum, "MANIFEST", bad)
    with pytest.raises(enum.EnumeratorMalfunction):
        enum.load_manifest()
    assert enum.main() == enum.EXIT_MALFUNCTION == 2
    assert "SchemaError" in capsys.readouterr().err


import ast as _ast


def _detect(src: str):
    tree = _ast.parse(src)
    ma, im, gn, gm = enum._collect_imports(tree)
    v = enum._LoadVisitor(ma, im, gn, gm)
    v.visit(tree)
    return [s[1] for s in v.sites]


@pytest.mark.parametrize("src,expect_kind", [
    ("import torch as t\ndef f(): return t.load('x')", "torch.load"),                 # alias
    ("from torch import load\ndef f(): return load('x')", "torch.load"),              # bare from-import
    ("from torch import load as L\ndef f(): return L('x')", "torch.load"),            # aliased from-import
    ("import pickle as p\ndef f(): return p.loads(b'x')", "pickle.loads"),            # pickle alias
    ("import joblib as jl\ndef f(): return jl.load('x')", "joblib.load"),             # joblib alias
    ("from transformers import AutoModel\ndef f(): return AutoModel.from_pretrained('x')", "from_pretrained"),
    ("from sentence_transformers import SentenceTransformer\ndef f(): return SentenceTransformer('x')", "ctor:SentenceTransformer"),
    ("import sentence_transformers as st\ndef f(): return st.SentenceTransformer('x')", "ctor:SentenceTransformer"),
    ("from paddleocr import PaddleOCR\ndef f(): return PaddleOCR()", "ctor:PaddleOCR"),
    ("import onnx\ndef f(): return onnx.load('m.onnx')", "onnx.load"),                # review 6
    ("import onnxruntime as ort\ndef f(): return ort.InferenceSession('m')", "ctor:InferenceSession"),
    ("from onnxruntime import InferenceSession\ndef f(): return InferenceSession('m')", "ctor:InferenceSession"),
])
def test_import_aware_detection_no_blind_spots(src: str, expect_kind: str) -> None:
    # review 5: a name-only matcher missed all of these; the import-aware detector must catch them.
    assert expect_kind in _detect(src), f"blind spot: {src!r} -> {_detect(src)}"


def test_real_hf_and_embedding_loaders_are_gated() -> None:
    # the specific loaders the review proved were MISSING must now be enumerated AND classified gated.
    manifest = enum.load_manifest()
    must_be_gated = {
        "src/core/ocr/providers/deepseek_hf.py": "from_pretrained/PaddleOCR (mounted /ocr)",
        "src/core/assistant/semantic_retrieval.py": "SentenceTransformer",
        "src/ml/embeddings/model.py": "SentenceTransformer",
    }
    for f in must_be_gated:
        hits = [e for k, e in manifest.items() if k.startswith(f + "::")]
        assert hits, f"{f} not enumerated (blind spot regressed)"
        assert all(e["class"] == "gated" and e.get("family") for e in hits), \
            f"{f} must be gated with a family: {hits}"


# --- C5 structural wiring check --------------------------------------------------------------------
# A `wired` raw deserializer (torch.load/pickle.load[s]/joblib.load) MUST reconstruct from
# activate_file/activate_bundle bytes; if the wrapper is removed and it reads bytes straight off a
# path again, the structural check must go RED (under enforce) — the remove-the-wrapper discriminator.
# Per the ratified W4 the check is present-but-advisory by default and BLOCKING only under
# ACTIVATION_ENFORCE_WIRING; both directions are exercised below.

# A LIVE gated raw loader that reconstructs from gateway-verified bytes (enclosing function delegates).
_WRAPPED_SRC = (
    "import io, torch\n"
    "from src.core.model_activation.activation_gateway import activate_file\n"
    "class M:\n"
    "    def load(self):\n"
    "        data = activate_file('fam/main', 'main')\n"
    "        if data is None:\n"
    "            return None\n"
    "        return torch.load(io.BytesIO(data), map_location='cpu')\n"
)
# The SAME site with the gateway wrapper REMOVED — a raw load straight off a path (unverified).
_UNWRAPPED_SRC = (
    "import torch\n"
    "class M:\n"
    "    def load(self, path):\n"
    "        return torch.load(path, map_location='cpu')\n"
)
# F2 DISCARD case — the enclosing function CALLS the gateway but DISCARDS the return, then loads a model
# straight off a filesystem path (self.model_path). The old presence-only wrap rule ("any enclosing
# function delegates to activate_*") scored this wrapped=True — a FALSE GREEN: an unverified path load
# marked as gateway-routed. The data-flow rule must see the loaded argument does not derive from the
# gateway bytes and score it wrapped=False -> a structural finding.
_DISCARD_SRC = (
    "import torch\n"
    "from src.core.model_activation.activation_gateway import activate_file\n"
    "class M:\n"
    "    def load(self):\n"
    "        data = activate_file('fam/main', 'main')  # called...\n"
    "        if data is None:\n"
    "            return None\n"
    "        return torch.load(self.model_path, map_location='cpu')  # ...but result DISCARDED\n"
)
_SITE_KEY = "src/fam.py::M.load::torch.load#0"


def _scan_with_manifest(tmp_path, monkeypatch, src: str, entry: dict, key: str = _SITE_KEY) -> None:
    """Point the enumerator at a one-file `src/` tree + a manifest classifying its single load site."""
    p = tmp_path / "src" / "fam.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(src, encoding="utf-8")
    mani = tmp_path / "manifest.json"
    mani.write_text(json.dumps({"sites": {key: entry}}), encoding="utf-8")
    monkeypatch.setattr(enum, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(enum, "SCAN_DIRS", ("src",))
    monkeypatch.setattr(enum, "MANIFEST", mani)


_WIRED_ENTRY = {"class": "gated", "family": "fam", "wiring": "wired", "reason": "live"}


def test_wrapped_wired_site_detected_as_wrapped(tmp_path, monkeypatch) -> None:
    # unit-level: the AST wrap detector sees the enclosing function delegate to activate_file.
    _scan_with_manifest(tmp_path, monkeypatch, _WRAPPED_SRC, _WIRED_ENTRY)
    found = enum.enumerate_sites()
    assert found[_SITE_KEY]["wrapped"] is True
    assert enum.structural_findings(found, enum.load_manifest()) == []


def test_wired_site_with_wrapper_present_is_green_even_enforced(tmp_path, monkeypatch) -> None:
    # positive control for the discriminator: wrapper present -> no finding, exit 0 even under enforce.
    _scan_with_manifest(tmp_path, monkeypatch, _WRAPPED_SRC, _WIRED_ENTRY)
    monkeypatch.setenv(enum.ENV_ENFORCE_WIRING, "1")
    assert enum.main() == enum.EXIT_OK == 0


def test_remove_the_wrapper_is_observed_RED_under_enforce(tmp_path, monkeypatch, capsys) -> None:
    # THE discriminator: same wired-marked site but the activate_file wrapper is REMOVED -> the raw
    # torch.load reads straight off a path -> structural inconsistency -> RED (exit 1) under enforce.
    _scan_with_manifest(tmp_path, monkeypatch, _UNWRAPPED_SRC, _WIRED_ENTRY)
    found = enum.enumerate_sites()
    assert found[_SITE_KEY]["wrapped"] is False  # detector no longer sees the gateway delegation
    monkeypatch.setenv(enum.ENV_ENFORCE_WIRING, "1")
    assert enum.main() == enum.EXIT_FINDING == 1
    err = capsys.readouterr().err
    assert "structural wiring, ENFORCED" in err and "wired-but-unwrapped" in err and _SITE_KEY in err


def test_f2_gateway_called_but_result_discarded_is_unwrapped(tmp_path, monkeypatch) -> None:
    # observed-RED against the OLD presence-only rule: the function DOES call activate_file, so the old
    # "any enclosing function delegates" test marked the raw torch.load(self.model_path) wrapped=True and
    # structural_findings returned [] — a false green for an unverified path load. The data-flow rule
    # sees the loaded argument (self.model_path) does not derive from the gateway bytes -> wrapped=False.
    _scan_with_manifest(tmp_path, monkeypatch, _DISCARD_SRC, _WIRED_ENTRY)
    found = enum.enumerate_sites()
    assert found[_SITE_KEY]["wrapped"] is False
    findings = enum.structural_findings(found, enum.load_manifest())
    assert [f for f in findings if f[0] == _SITE_KEY and f[1] == "wired-but-unwrapped"], findings


def test_f2_discard_case_is_observed_RED_under_enforce(tmp_path, monkeypatch, capsys) -> None:
    # end-to-end: under enforce the F2 discard case reds CI (exit 1) with a wired-but-unwrapped finding.
    _scan_with_manifest(tmp_path, monkeypatch, _DISCARD_SRC, _WIRED_ENTRY)
    monkeypatch.setenv(enum.ENV_ENFORCE_WIRING, "1")
    assert enum.main() == enum.EXIT_FINDING == 1
    err = capsys.readouterr().err
    assert "structural wiring, ENFORCED" in err and "wired-but-unwrapped" in err and _SITE_KEY in err


# F2 ROUND-2 — GATEWAY-CALLED-AFTER-THE-LOADER. The wrap check must be DOMINANCE / lexical-order aware:
# a gateway-derived binding counts for a load site ONLY if it lexically precedes (dominates) that site.
# Here the raw torch.load reads the raw `path` PARAM, and the gateway rebind of `path` happens on the
# line AFTER the loader — so the value the loader deserializes is the unverified param, not gateway bytes.
# observed-RED against the OLD order-blind rule: _collect_gateway_vars scanned the WHOLE function body and
# admitted `path` as a gateway var regardless of position, scoring wrapped=True (a false green). The
# dominance rule rejects a binding at a lineno >= the load site -> wrapped=False -> a wired-but-unwrapped
# finding (exit 1 under enforce).
_GATEWAY_AFTER_LOADER_SRC = (
    "import io, torch\n"
    "from src.core.model_activation.activation_gateway import activate_file\n"
    "class M:\n"
    "    def load(self, path):\n"
    "        loaded = torch.load(path, map_location='cpu')  # unverified raw-path load\n"
    "        path = activate_file('x', 'main')  # gateway binding AFTER the loader -> must NOT count\n"
    "        return loaded\n"
)


def test_gateway_bound_after_loader_is_unwrapped(tmp_path, monkeypatch) -> None:
    # unit-level: the loaded arg (`path`) has its only gateway binding on a LATER line, so the
    # dominance-aware detector scores it wrapped=False and structural_findings flags it.
    _scan_with_manifest(tmp_path, monkeypatch, _GATEWAY_AFTER_LOADER_SRC, _WIRED_ENTRY)
    found = enum.enumerate_sites()
    assert found[_SITE_KEY]["wrapped"] is False
    findings = enum.structural_findings(found, enum.load_manifest())
    assert [f for f in findings if f[0] == _SITE_KEY and f[1] == "wired-but-unwrapped"], findings


def test_gateway_bound_after_loader_is_observed_RED_under_enforce(tmp_path, monkeypatch, capsys) -> None:
    # end-to-end: under enforce the gateway-after-loader case reds CI (exit 1) with wired-but-unwrapped.
    _scan_with_manifest(tmp_path, monkeypatch, _GATEWAY_AFTER_LOADER_SRC, _WIRED_ENTRY)
    monkeypatch.setenv(enum.ENV_ENFORCE_WIRING, "1")
    assert enum.main() == enum.EXIT_FINDING == 1
    err = capsys.readouterr().err
    assert "structural wiring, ENFORCED" in err and "wired-but-unwrapped" in err and _SITE_KEY in err


def test_remove_the_wrapper_is_advisory_only_by_default(tmp_path, monkeypatch, capsys) -> None:
    # W4: without the enforce flag the SAME inconsistency is advisory — printed, but exit 0 (non-blocking),
    # so it cannot red CI while in-scope families are still being wired.
    _scan_with_manifest(tmp_path, monkeypatch, _UNWRAPPED_SRC, _WIRED_ENTRY)
    monkeypatch.delenv(enum.ENV_ENFORCE_WIRING, raising=False)
    assert enum.main() == enum.EXIT_OK == 0
    err = capsys.readouterr().err
    assert "ADVISORY (structural wiring, non-blocking" in err and "wired-but-unwrapped" in err


def test_gate_before_wired_unwrapped_raw_loader_is_consistent(tmp_path, monkeypatch) -> None:
    # a LIVE gated raw loader not yet routed (gate-before-wired) is a raw load today -> consistent,
    # green even under enforce (this is why enforce can be flipped without redding deferred families).
    entry = {"class": "gated", "family": "fam", "wiring": "gate-before-wired", "reason": "deferred"}
    _scan_with_manifest(tmp_path, monkeypatch, _UNWRAPPED_SRC, entry)
    monkeypatch.setenv(enum.ENV_ENFORCE_WIRING, "1")
    assert enum.main() == enum.EXIT_OK == 0


def test_gate_before_wired_but_actually_wrapped_reds_under_enforce(tmp_path, monkeypatch, capsys) -> None:
    # the inverse lie: a site marked gate-before-wired that IS routed through the gateway -> stale
    # marker -> RED under enforce (forces reclassification to wired, so the manifest cannot under-claim).
    entry = {"class": "gated", "family": "fam", "wiring": "gate-before-wired", "reason": "stale"}
    _scan_with_manifest(tmp_path, monkeypatch, _WRAPPED_SRC, entry)
    monkeypatch.setenv(enum.ENV_ENFORCE_WIRING, "1")
    assert enum.main() == enum.EXIT_FINDING == 1
    assert "unwired-but-wrapped" in capsys.readouterr().err


def test_gated_raw_loader_missing_wiring_reds_under_enforce(tmp_path, monkeypatch, capsys) -> None:
    # a gated site with no wiring lifecycle is itself a finding under enforce — the lifecycle must be
    # declared so a new gated load cannot slip in unrouted-and-unmarked.
    entry = {"class": "gated", "family": "fam", "reason": "no wiring field"}
    _scan_with_manifest(tmp_path, monkeypatch, _WRAPPED_SRC, entry)
    monkeypatch.setenv(enum.ENV_ENFORCE_WIRING, "1")
    assert enum.main() == enum.EXIT_FINDING == 1
    assert "missing-wiring" in capsys.readouterr().err


def test_real_tree_is_structurally_consistent_under_enforce() -> None:
    # integration: the real source tree + manifest carry NO structural inconsistency — every `wired`
    # raw loader actually reconstructs from activate_file/activate_bundle bytes, and every
    # gate-before-wired/latent raw loader is still a raw load. So the owner can flip enforce ON safely.
    found = enum.enumerate_sites()
    manifest = enum.load_manifest()
    assert enum.structural_findings(found, manifest) == []


def test_every_gated_site_carries_a_valid_wiring() -> None:
    # manifest self-consistency: all 38 gated sites declare a wiring lifecycle in the valid set.
    manifest = enum.load_manifest()
    bad = {k: e.get("wiring") for k, e in manifest.items()
           if e["class"] == "gated" and e.get("wiring") not in enum.VALID_WIRING}
    assert not bad, f"gated sites with missing/invalid wiring: {bad}"


# --- C5 round-3 (1): outer-scope gateway bindings do NOT dominate a nested-function load -----------
# The old code inherited every parent-scope gateway var into a nested function scope RELAXED to
# (0, ()) so it ALWAYS dominated — a raw loader inside a closure was scored wrapped=True merely
# because an OUTER function bound a gateway var of the same name. That is a false green: without real
# cross-scope call-order analysis the enumerator cannot prove the outer binding runs BEFORE the nested
# load (the closure may be called first, or read the raw param). The conservative fail-closed rule:
# a raw loader inside a nested function/lambda is wrapped ONLY if a gateway binding WITHIN THAT SAME
# function dominates it — an outer binding never counts.
#
# Repro: `load_outer(path)` binds `path = activate_file(...)`, but the nested `inner` closure loads the
# free var `path` (its gateway rebind in the enclosing scope may not have run when `inner` executes).
_CLOSURE_BEFORE_OUTER_SRC = (
    "import torch\n"
    "from src.core.model_activation.activation_gateway import activate_file\n"
    "def load_outer(path):\n"
    "    def inner():\n"
    "        return torch.load(path, map_location='cpu')  # loads a free var — outer binding must NOT count\n"
    "    path = activate_file('x', 'main')  # enclosing-scope binding — cannot dominate the nested load\n"
    "    return inner\n"
)
_CLOSURE_SITE_KEY = "src/fam.py::load_outer.inner::torch.load#0"


def test_closure_load_not_wrapped_by_outer_gateway_binding(tmp_path, monkeypatch) -> None:
    # observed-RED against the OLD inherit-and-relax rule: the outer `path = activate_file(...)` binding
    # was inherited into `inner` as (0, ()) and always dominated -> wrapped=True, structural_findings==[]
    # (a false green for an unverified free-var load). The no-inheritance rule sees `inner` has NO
    # in-scope gateway binding -> wrapped=False -> a wired-but-unwrapped finding.
    _scan_with_manifest(tmp_path, monkeypatch, _CLOSURE_BEFORE_OUTER_SRC, _WIRED_ENTRY, key=_CLOSURE_SITE_KEY)
    found = enum.enumerate_sites()
    assert found[_CLOSURE_SITE_KEY]["wrapped"] is False
    findings = enum.structural_findings(found, enum.load_manifest())
    assert [f for f in findings if f[0] == _CLOSURE_SITE_KEY and f[1] == "wired-but-unwrapped"], findings


def test_closure_load_before_outer_gateway_is_observed_RED_under_enforce(tmp_path, monkeypatch, capsys) -> None:
    # end-to-end: under enforce the nested closure load reds CI (exit 1) with a wired-but-unwrapped finding.
    _scan_with_manifest(tmp_path, monkeypatch, _CLOSURE_BEFORE_OUTER_SRC, _WIRED_ENTRY, key=_CLOSURE_SITE_KEY)
    monkeypatch.setenv(enum.ENV_ENFORCE_WIRING, "1")
    assert enum.main() == enum.EXIT_FINDING == 1
    err = capsys.readouterr().err
    assert "structural wiring, ENFORCED" in err and "wired-but-unwrapped" in err and _CLOSURE_SITE_KEY in err


# --- C5 round-3 (2): only a call resolving to the CANONICAL gateway module counts ------------------
# The old `_is_activation_call` matched by NAME alone (`activate_file` / `activate_bundle`), so a
# project-local FAKE or shadowed `def activate_file(...)` counted as the real gateway — a raw load
# wrapping its bytes was scored wrapped=True though nothing verified them. The fix resolves the call
# against the file's import table: it counts only if `activate_file`/`activate_bundle` was imported
# from src.core.model_activation.activation_gateway (bare, aliased, or attribute form). A locally
# defined or differently-sourced name does NOT.
_FAKE_GATEWAY_SRC = (
    "import io, torch\n"
    "def activate_file(artifact_id, family):  # project-local FAKE — NOT the canonical gateway\n"
    "    with open(artifact_id, 'rb') as fh:\n"
    "        return fh.read()\n"
    "class M:\n"
    "    def load(self):\n"
    "        data = activate_file('fam/main', 'main')\n"
    "        return torch.load(io.BytesIO(data), map_location='cpu')\n"
)


def test_local_fake_activate_file_is_not_the_gateway(tmp_path, monkeypatch) -> None:
    # observed-RED against the OLD name-only matcher: `activate_file` matched by name, so `data` became a
    # gateway var and `torch.load(io.BytesIO(data))` scored wrapped=True, structural_findings==[] (a false
    # green — the bytes came from an unverified local read). Import-table resolution sees the name does not
    # resolve to the canonical gateway module -> not a gateway call -> wrapped=False -> a finding.
    _scan_with_manifest(tmp_path, monkeypatch, _FAKE_GATEWAY_SRC, _WIRED_ENTRY)
    found = enum.enumerate_sites()
    assert found[_SITE_KEY]["wrapped"] is False
    findings = enum.structural_findings(found, enum.load_manifest())
    assert [f for f in findings if f[0] == _SITE_KEY and f[1] == "wired-but-unwrapped"], findings


def test_local_fake_activate_file_is_observed_RED_under_enforce(tmp_path, monkeypatch, capsys) -> None:
    # end-to-end: under enforce the fake-gateway wrap reds CI (exit 1) with a wired-but-unwrapped finding.
    _scan_with_manifest(tmp_path, monkeypatch, _FAKE_GATEWAY_SRC, _WIRED_ENTRY)
    monkeypatch.setenv(enum.ENV_ENFORCE_WIRING, "1")
    assert enum.main() == enum.EXIT_FINDING == 1
    err = capsys.readouterr().err
    assert "structural wiring, ENFORCED" in err and "wired-but-unwrapped" in err and _SITE_KEY in err


# --- C5 round-4: gateway-name resolution must be SCOPE-AWARE (shadowing fails closed) ---------------
# The canonical import IS present at file level (so `activate_file` lands in gw_names via ast.walk), but
# a function SHADOWS that name — with a PARAMETER (A) or a LOCAL `def` (B) — so within that scope the
# name is NOT the canonical gateway. The old file-level gw_names + name-only `_is_activation_call` scored
# the shadowed call as canonical, laundering an unverified value into a gateway var: `data` derives from
# the SHADOWED activate_file, so `torch.load(io.BytesIO(data))` scored wrapped=True — a FALSE GREEN. The
# scope-aware fix subtracts shadowed gateway names from the effective gw set for that scope (and nested
# scopes), fail-closed: a shadowing param/def/assign means the name is not the gateway there ->
# wrapped=False -> a wired-but-unwrapped finding. NB the loaded value must DERIVE from the shadowed call
# (mirror _WRAPPED_SRC), else the load reads a bare path and is already unwrapped on the old code — which
# would not be an observed-RED for THIS shadowing bug.
_PARAM_SHADOW_SRC = (
    "import io, torch\n"
    "from src.core.model_activation.activation_gateway import activate_file\n"
    "class M:\n"
    "    def load(self, activate_file):  # PARAM shadows the canonical import within this scope\n"
    "        data = activate_file('fam/main', 'main')  # NOT the gateway — it is the param\n"
    "        return torch.load(io.BytesIO(data), map_location='cpu')\n"
)
_LOCAL_DEF_SHADOW_SRC = (
    "import io, torch\n"
    "from src.core.model_activation.activation_gateway import activate_file\n"
    "class M:\n"
    "    def load(self):\n"
    "        def activate_file(a, b):  # LOCAL def shadows the canonical import within this scope\n"
    "            return None\n"
    "        data = activate_file('fam/main', 'main')  # NOT the gateway — it is the local def\n"
    "        return torch.load(io.BytesIO(data), map_location='cpu')\n"
)


def test_param_shadow_of_activate_file_is_not_the_gateway(tmp_path, monkeypatch) -> None:
    # observed-RED against the OLD file-level gw_names: the param `activate_file` shadows the canonical
    # import, but the old name-only matcher still counted `activate_file(...)` as the gateway, so `data`
    # became a gateway var and `torch.load(io.BytesIO(data))` scored wrapped=True, structural_findings==[]
    # (a false green). The scope-aware rule drops the shadowed name -> not a gateway call -> wrapped=False.
    _scan_with_manifest(tmp_path, monkeypatch, _PARAM_SHADOW_SRC, _WIRED_ENTRY)
    found = enum.enumerate_sites()
    assert found[_SITE_KEY]["wrapped"] is False
    findings = enum.structural_findings(found, enum.load_manifest())
    assert [f for f in findings if f[0] == _SITE_KEY and f[1] == "wired-but-unwrapped"], findings


def test_param_shadow_of_activate_file_is_observed_RED_under_enforce(tmp_path, monkeypatch, capsys) -> None:
    # end-to-end: under enforce the param-shadow case reds CI (exit 1) with a wired-but-unwrapped finding.
    _scan_with_manifest(tmp_path, monkeypatch, _PARAM_SHADOW_SRC, _WIRED_ENTRY)
    monkeypatch.setenv(enum.ENV_ENFORCE_WIRING, "1")
    assert enum.main() == enum.EXIT_FINDING == 1
    err = capsys.readouterr().err
    assert "structural wiring, ENFORCED" in err and "wired-but-unwrapped" in err and _SITE_KEY in err


def test_local_def_shadow_of_activate_file_is_not_the_gateway(tmp_path, monkeypatch) -> None:
    # observed-RED against the OLD file-level gw_names: a local `def activate_file` shadows the canonical
    # import, yet the old name-only matcher counted the shadowed call as the gateway, so `data` became a
    # gateway var and `torch.load(io.BytesIO(data))` scored wrapped=True, structural_findings==[] (a false
    # green). The scope-aware rule drops the locally-defined name -> not a gateway call -> wrapped=False.
    _scan_with_manifest(tmp_path, monkeypatch, _LOCAL_DEF_SHADOW_SRC, _WIRED_ENTRY)
    found = enum.enumerate_sites()
    assert found[_SITE_KEY]["wrapped"] is False
    findings = enum.structural_findings(found, enum.load_manifest())
    assert [f for f in findings if f[0] == _SITE_KEY and f[1] == "wired-but-unwrapped"], findings


def test_local_def_shadow_of_activate_file_is_observed_RED_under_enforce(tmp_path, monkeypatch, capsys) -> None:
    # end-to-end: under enforce the local-def-shadow case reds CI (exit 1) with a wired-but-unwrapped finding.
    _scan_with_manifest(tmp_path, monkeypatch, _LOCAL_DEF_SHADOW_SRC, _WIRED_ENTRY)
    monkeypatch.setenv(enum.ENV_ENFORCE_WIRING, "1")
    assert enum.main() == enum.EXIT_FINDING == 1
    err = capsys.readouterr().err
    assert "structural wiring, ENFORCED" in err and "wired-but-unwrapped" in err and _SITE_KEY in err
