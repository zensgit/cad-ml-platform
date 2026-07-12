#!/usr/bin/env python3
"""Track E slice-1: evaluation-integrity splitter + artifact builder (evaluation-integrity-v2).

Produces the versioned, reproducible evaluation-integrity artifact that the L3 gate
(``scripts/eval_integrity_gate.py``) requires before ``scripts/auto_retrain.sh`` may mutate or
train. The integrity core (PRODUCT_STRATEGY.md §8.1):

  * **content-hash** — identical file CONTENT is detected regardless of path. Unreadable bytes
    **fail closed** (the row is quarantined, never silently treated as "distinct" — which would
    let identical content leak across the train/holdout boundary).
  * **normalized-family** — augmentation / revision variants of one drawing collapse to one family
    (§8.1.1), and any two families sharing identical content are unioned into one split unit, so a
    drawing (and its duplicates) can never straddle train and holdout (§8.1.3).
  * **conflict quarantine** — identical content carrying inconsistent labels is excluded (§8.1.2).
  * **deterministic split_digest** — re-deriving the split from the same manifest yields the same
    digest, so ``verify`` goes RED when a split is tampered or duplicate content is reintroduced
    (§8.1 exit condition). This reproducibility check is wired **dry-run first** (§8.1.7), not
    blocking.

Slice-1 does NOT compute model metrics (per-class / calibration / false-duplicate / missed-reuse
need the model run over the holdout); ``build_artifact`` takes them from an eval-results file. The
gate's contract constants are IMPORTED here, so the artifact is gate-conformant by construction
(single source of truth). Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Optional manifest columns that, when present and non-empty, give an AUTHORITATIVE family/source
# grouping (§8.1.6 versioned manifest carries `family`). Preferred over the filename heuristic.
_FAMILY_COLUMNS = ("family", "source_id", "source_drawing", "drawing_id")

# Resolve the sibling gate whether this file is imported as ``scripts.track_e_eval_integrity``
# (pytest, repo root on sys.path) or run as ``python3 scripts/track_e_eval_integrity.py`` (CLI,
# scripts/ on sys.path). Add the scripts/ dir so the top-level import works in both.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# Single source of truth for the artifact contract — imported, never re-hardcoded, so the producer
# can never drift from the gate that consumes it.
from eval_integrity_gate import (  # noqa: E402  (sibling script; scripts/ ensured on sys.path above)
    REQUIRED_METRIC_KEYS,
    REQUIRED_SPLIT_STRATEGY,
    REQUIRED_VERSION,
    validate_artifact,
)

# Augmentation / revision / copy markers that denote the SAME source drawing. Stripped (repeatedly)
# from a path stem to collapse variants to one family. Conservative on purpose: over-collapsing is
# safe (keeps more content together), under-collapsing is the leakage risk.
import re

_VARIANT_MARKERS = re.compile(
    r"("
    r"_aug\d*"                 # augmentation copies
    r"|_rot-?\d+"              # rotation
    r"|_flip[a-z]*"            # flip/mirror
    r"|_scale[\d._]*"          # scaling
    r"|_noise\d*"
    r"|_jitter\d*"
    r"|_v\d+"                  # version
    r"|_rev\d+"                # revision
    r"|[-_ ]?copy\d*"          # "_copy", " - copy" (Windows), "copy2"
    r"| ?\(\d+\)"              # OS duplicate "gear (1)" / "gear(1)"
    r"|[-_ ]\d+"               # separated copy index (foo-1, foo_2, "foo 3")
    r"|\d+"                    # bare trailing digit run (gear2 -> gear); errs toward collapse
    r")+$",
    re.IGNORECASE,
)

_HOLDOUT_SALT = "evaluation-integrity-v2"
DEFAULT_HOLDOUT_FRACTION = 0.2


class IntegrityError(Exception):
    """A fail-closed integrity violation (unreadable content, malformed manifest, bad metrics)."""


class QuarantineRow(Exception):
    """Signals a row must be quarantined rather than split (raised internally, not fatal)."""


def normalized_family(file_path: str) -> str:
    """Collapse a path to its source-drawing family key (variants → one family).

    Heuristic FALLBACK used only when the manifest has no explicit family column. Unicode-NFC
    normalized (so NFC/NFD spellings of the same name collapse) then variant markers are stripped.
    It errs toward OVER-collapse (bare trailing digits included): for a leakage guard over-collapse
    is safe (keeps a drawing's variants together, costs holdout diversity), under-collapse leaks.
    """
    base = unicodedata.normalize("NFC", Path(str(file_path)).stem).lower()
    stem = base
    prev = None
    while stem != prev:
        prev = stem
        stem = _VARIANT_MARKERS.sub("", stem).strip(" -_")
    return stem or base


def _family_key(row: dict) -> str:
    """Authoritative family from an explicit manifest column if present (§8.1.6), else the
    filename heuristic. Prefixed so a filename can never collide with a declared family id."""
    for col in _FAMILY_COLUMNS:
        val = str(row.get(col, "")).strip()
        if val:
            return f"declared:{unicodedata.normalize('NFC', val).lower()}"
    return f"file:{normalized_family(str(row.get('file_path', '')))}"


def content_hash(file_path: str, *, root: Optional[Path] = None) -> str:
    """sha256 of the file's bytes. Fail-closed: unreadable → QuarantineRow (never 'distinct')."""
    p = Path(file_path)
    if root is not None and not p.is_absolute():
        p = root / p
    try:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, ValueError) as exc:
        # OSError: missing/permission. ValueError: embedded NUL byte in the path (survives CSV
        # ingestion). Both fail closed -> quarantine, never silently "distinct".
        raise QuarantineRow(f"unreadable content for {file_path!r}: {exc}") from exc


class _Union:
    """Union-find over split units (families joined when they share identical content)."""

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic: smaller key becomes root
            hi, lo = (ra, rb) if ra > rb else (rb, ra)
            self._parent[hi] = lo


def _read_manifest(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise IntegrityError(f"manifest {path!r} is empty")
    required = {"file_path", "taxonomy_v2_class"}
    missing = required - set(rows[0].keys())
    if missing:
        raise IntegrityError(f"manifest {path!r} missing columns: {', '.join(sorted(missing))}")
    return rows


def compute_split(
    rows: Iterable[dict],
    *,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    root: Optional[Path] = None,
) -> dict:
    """Assign each row to 'train' or 'holdout' by leakage-safe component, with conflict quarantine.

    Split unit = a connected component of (normalized-family ∪ identical-content). Identical content
    (same content-hash) with inconsistent labels is quarantined. Deterministic: component → bucket
    by hashing the canonical component key, so the same manifest always yields the same split.
    """
    rows = list(rows)
    hash_to_label: Dict[str, str] = {}
    conflict_hashes: set = set()
    quarantined: List[dict] = []
    prepared: List[Tuple[dict, str, str]] = []  # (row, family, content_hash)
    uf = _Union()

    for row in rows:
        fp = str(row.get("file_path", "")).strip()
        label = str(row.get("taxonomy_v2_class", "")).strip()
        if not fp or not label:
            quarantined.append({"file_path": fp, "reason": "missing file_path or label"})
            continue
        try:
            ch = content_hash(fp, root=root)
        except QuarantineRow as exc:
            quarantined.append({"file_path": fp, "reason": str(exc)})
            continue
        fam = _family_key(row)  # authoritative manifest column if present, else filename heuristic
        # identical content with a DIFFERENT label anywhere -> conflict (§8.1.2)
        seen = hash_to_label.get(ch)
        if seen is not None and seen != label:
            conflict_hashes.add(ch)
        else:
            hash_to_label.setdefault(ch, label)
        uf.union(f"fam:{fam}", f"content:{ch}")  # families sharing content collapse into one unit
        prepared.append((row, fam, ch))

    assignment: Dict[str, str] = {}
    component_of: Dict[str, str] = {}
    for row, fam, ch in prepared:
        if ch in conflict_hashes:
            quarantined.append({"file_path": str(row["file_path"]), "reason": f"label-conflict content {ch[:12]}"})
            continue
        comp = uf.find(f"fam:{fam}")
        component_of[str(row["file_path"])] = comp

    # deterministic bucket per component
    comp_side: Dict[str, str] = {}
    for comp in sorted(set(component_of.values())):
        digest = hashlib.sha256(f"{_HOLDOUT_SALT}|{comp}".encode()).hexdigest()
        frac = int(digest[:16], 16) / float(1 << 64)
        comp_side[comp] = "holdout" if frac < holdout_fraction else "train"

    for fp, comp in component_of.items():
        assignment[fp] = comp_side[comp]

    return {
        "assignment": assignment,               # file_path -> 'train'|'holdout'
        "components": len(set(component_of.values())),
        "quarantined": quarantined,
        "holdout_fraction": holdout_fraction,
    }


def split_digest(split: dict) -> str:
    """A stable digest of the split assignment; changes iff any row's side changes."""
    items = sorted(split["assignment"].items())
    payload = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_artifact(
    rows: Iterable[dict],
    metrics: dict,
    *,
    label_authority: str = "manifest:taxonomy_v2_class",
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    root: Optional[Path] = None,
) -> dict:
    """Assemble a gate-conformant evaluation-integrity-v2 artifact. Metrics come from the eval run."""
    if not isinstance(metrics, dict):
        raise IntegrityError("metrics must be an object carrying the §8.1.4 families")
    missing = [k for k in REQUIRED_METRIC_KEYS if k not in metrics]
    if missing:
        raise IntegrityError(f"metrics missing required families (§8.1.4): {', '.join(missing)}")

    split = compute_split(rows, holdout_fraction=holdout_fraction, root=root)
    sides = list(split["assignment"].values())
    holdout_n = sum(1 for s in sides if s == "holdout")

    artifact = {
        "schema_version": REQUIRED_VERSION,
        "split_strategy": REQUIRED_SPLIT_STRATEGY,
        "holdout": {
            "type": "content-hash+normalized-family component",
            "fraction": holdout_fraction,
            "components": split["components"],
            "holdout_rows": holdout_n,
            "train_rows": len(sides) - holdout_n,
            "quarantined": len(split["quarantined"]),
        },
        "metrics": metrics,
        "label_authority": label_authority,
        "reproducible": True,
        "split_digest": split_digest(split),
    }
    # gate-conformant by construction — assert against the imported contract before returning
    _assert_gate_conformant(artifact)
    return artifact


def _assert_gate_conformant(artifact: dict) -> None:
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(artifact, f)
        tmp = f.name
    try:
        validate_artifact(tmp)  # raises GateError if the producer ever drifts from the gate
    finally:
        Path(tmp).unlink(missing_ok=True)


def verify_reproducible(rows: Iterable[dict], artifact: dict, *, root: Optional[Path] = None) -> None:
    """Re-derive the split and confirm the artifact's digest matches. RED on any drift.

    This is the §8.1 exit-condition check ("changing a split → red"), wired dry-run first (§8.1.7).
    """
    frac = artifact.get("holdout", {}).get("fraction", DEFAULT_HOLDOUT_FRACTION)
    recomputed = split_digest(compute_split(rows, holdout_fraction=frac, root=root))
    stored = artifact.get("split_digest")
    if stored != recomputed:
        raise IntegrityError(
            "reproducibility check FAILED: the split changed since the artifact was built "
            f"(stored {str(stored)[:12]} != recomputed {recomputed[:12]}). A split was tampered, "
            "the manifest content changed, or duplicate content was reintroduced."
        )


# --- CLI ----------------------------------------------------------------------------------------
def _load_metrics(path: Optional[str]) -> dict:
    if not path:
        # placeholder metrics carry every §8.1.4 family so the shape is exercised; a REAL run
        # overwrites these. Zeros are explicit, not silently "good".
        return {k: ({} if k == "per_class" else 0.0) for k in REQUIRED_METRIC_KEYS}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Track E evaluation-integrity splitter/artifact.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split", help="print the leakage-safe split summary")
    sp.add_argument("--manifest", required=True)
    sp.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)

    bp = sub.add_parser("build", help="build the evaluation-integrity-v2 artifact")
    bp.add_argument("--manifest", required=True)
    bp.add_argument("--metrics", help="eval-results JSON with the §8.1.4 metric families")
    bp.add_argument("--out", required=True)
    bp.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)

    vp = sub.add_parser("verify", help="dry-run reproducibility check (§8.1.7): split unchanged?")
    vp.add_argument("--manifest", required=True)
    vp.add_argument("--artifact", required=True)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "split":
            split = compute_split(_read_manifest(args.manifest), holdout_fraction=args.holdout_fraction)
            print(json.dumps({
                "components": split["components"],
                "quarantined": len(split["quarantined"]),
                "digest": split_digest(split),
            }, indent=2))
            return 0
        if args.cmd == "build":
            rows = _read_manifest(args.manifest)
            artifact = build_artifact(rows, _load_metrics(args.metrics), holdout_fraction=args.holdout_fraction)
            Path(args.out).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
            print(f"wrote {args.out} (digest {artifact['split_digest'][:12]})")
            return 0
        if args.cmd == "verify":
            rows = _read_manifest(args.manifest)
            artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
            verify_reproducible(rows, artifact)
            print("reproducibility check PASS (dry-run): split matches the artifact digest")
            return 0
    except IntegrityError as exc:
        sys.stderr.write(f"[track-e] DRY-RUN reproducibility/integrity finding: {exc}\n")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
