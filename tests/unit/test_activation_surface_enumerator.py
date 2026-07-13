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


def test_invalid_class_is_rejected(tmp_path, monkeypatch) -> None:
    real = json.loads(enum.MANIFEST.read_text(encoding="utf-8"))
    k = sorted(real["sites"])[0]
    real["sites"][k] = {"class": "totally-safe-trust-me", "reason": "x"}
    bad = tmp_path / "activation_surface.json"
    bad.write_text(json.dumps(real), encoding="utf-8")
    monkeypatch.setattr(enum, "MANIFEST", bad)
    with pytest.raises(SystemExit):
        enum.load_manifest()


def test_every_gated_site_carries_a_family() -> None:
    # a gated (production-reachable) site must name its model family, so the membrane can bind it.
    manifest = enum.load_manifest()
    missing = [k for k, e in manifest.items() if e["class"] == "gated" and not e.get("family")]
    assert not missing, f"gated sites without a family: {missing}"
