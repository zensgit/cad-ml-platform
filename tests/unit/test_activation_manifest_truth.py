"""Regression guard for the L3 activation-surface manifest's output truth.

The manifest ``_doc`` and per-site ``reason`` strings, plus the enumerator source, were
corrected (PR #521 for the enumerator output, and the manifest-truth PR for the JSON data)
to a *conservatively-classified AST* framing under owner decision **(b)**: a ``gated`` site
is not asserted "production-reachable" (per-site logical reachability is a Wave-1 audit), and
it is fixed-hash- / bundle-digest-checked, not "frozen" (the rejected option (a) hard-refuse).

This test forbids either the over-claimed reachability label or the rejected-(a) hard-refuse
wording from regressing into the machine inventory or the enumerator source.
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "scripts" / "ci" / "activation_surface.json"
ENUMERATOR = ROOT / "scripts" / "ci" / "activation_surface_enumerator.py"

# Over-claimed reachability label + rejected option (a) hard-refuse wording.
FORBIDDEN = ("production-reachable", "must be frozen")

# The single LIVE (non-latent) pickle-classifier load site. It is lazy-first-predict,
# NOT a startup load — see the pinpoint guard below.
# Re-keyed pickle.load#0 -> pickle.loads#0: the C2 wiring routed load_model through the activation
# gateway, so the raw idiom changed from pickle.load(f) (path) to pickle.loads(data) (verified bytes).
# The guard's intent is unchanged — same site, still lazy-first-predict, still no-startup.
LIVE_PICKLE_SITE = "src/ml/classifier.py::load_model::pickle.loads#0"


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


def test_live_pickle_load_site_is_lazy_first_predict_not_startup() -> None:
    """Pinpoint guard: the LIVE classifier pickle load is lazy-first-predict, NOT a startup load.

    ``classifier.py::load_model`` (``src/ml/classifier.py:47``) is invoked ONLY from ``predict``
    (``src/ml/classifier.py:124``); there is no module-import / app-startup call. An earlier manifest
    reason read "LIVE startup / lazy-first-predict", over-stating a startup activation path that does
    not exist in the source. "startup" is NOT a blanket-forbidden word (a genuinely startup-loaded
    site could legitimately use it), so this guard is scoped to THIS one site — it must never
    re-acquire the "startup" claim, and must keep naming the real lazy-first-predict path.
    """
    sites = json.loads(MANIFEST.read_text(encoding="utf-8"))["sites"]
    assert LIVE_PICKLE_SITE in sites, f"{LIVE_PICKLE_SITE} missing from manifest"
    entry = sites[LIVE_PICKLE_SITE]
    reason = entry["reason"].lower()
    assert entry["class"] == "gated" and entry.get("family") == "pickle-classifier", (
        f"{LIVE_PICKLE_SITE} must stay a gated pickle-classifier site: {entry!r}"
    )
    assert "startup" not in reason, (
        f"{LIVE_PICKLE_SITE} reason re-acquired the 'startup' claim: {entry['reason']!r}; "
        "load_model (classifier.py:47) is called only from predict (classifier.py:124) — "
        "lazy-first-predict, no startup/import-time load."
    )
    assert "lazy-first-predict" in reason, (
        f"{LIVE_PICKLE_SITE} reason must still name the real lazy-first-predict path: "
        f"{entry['reason']!r}"
    )
