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
    # a gated (production-reachable) site must name its model family, so the membrane can bind it.
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


import ast as _ast


def _detect(src: str):
    tree = _ast.parse(src)
    ma, im = enum._collect_imports(tree)
    v = enum._LoadVisitor(ma, im)
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
