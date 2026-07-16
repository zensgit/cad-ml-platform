"""Regression guard for the L3 activation-surface manifest's output truth.

The manifest ``_doc`` and per-site ``reason`` strings, plus the enumerator source, were
corrected (PR #521 for the enumerator output, and the manifest-truth PR for the JSON data)
to a *conservatively-classified AST* framing under owner decision **(b)**: a ``gated`` site
is not asserted "production-reachable" (per-site logical reachability is a Wave-1 audit), and
it is fixed-hash- / bundle-digest-checked, not "frozen" (the rejected option (a) hard-refuse).

This test forbids either the over-claimed reachability label or the rejected-(a) hard-refuse
wording from regressing into the machine inventory or the enumerator source.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "scripts" / "ci" / "activation_surface.json"
ENUMERATOR = ROOT / "scripts" / "ci" / "activation_surface_enumerator.py"

# Over-claimed reachability label + rejected option (a) hard-refuse wording.
FORBIDDEN = ("production-reachable", "must be frozen")


def _forbidden_hits(path: pathlib.Path) -> list:
    text = path.read_text(encoding="utf-8").lower()
    return [phrase for phrase in FORBIDDEN if phrase in text]


def test_manifest_has_no_rejected_a_or_overclaimed_reachability() -> None:
    hits = _forbidden_hits(MANIFEST)
    assert not hits, (
        f"{MANIFEST.name} reintroduced rejected-(a) / over-claimed wording {hits}; "
        "`gated` sites are conservatively classified (per-site reachability is a Wave-1 audit) "
        "and are fixed-hash-/bundle-digest-checked under owner decision (b), never 'frozen' and "
        "never asserted 'production-reachable'."
    )


def test_enumerator_source_has_no_rejected_a_or_overclaimed_reachability() -> None:
    hits = _forbidden_hits(ENUMERATOR)
    assert not hits, (
        f"{ENUMERATOR.name} reintroduced rejected-(a) / over-claimed wording {hits}."
    )
