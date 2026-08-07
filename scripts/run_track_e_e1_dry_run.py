#!/usr/bin/env python3
"""Operator entry for Track E Slice E1 dry-run (torch-free).

Runs leakage-safe split + versioned manifest build + verify against the
ratified design-lock (#531) implementation on main (#542). Never imports
``eval_integrity_gate`` and never unlocks retraining.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow ``python scripts/run_track_e_e1_dry_run.py`` from repo root.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import track_e_eval_integrity as split_mod  # noqa: E402
import track_e_manifest as man_mod  # noqa: E402


def run_dry_run(
    *,
    manifest_csv: Path,
    root: Path | None,
    out_dir: Path,
    holdout_fraction: float,
    source: str = "dry-run:operator",
    license_: str = "internal-review-only",
    label_authority: str = "manifest:taxonomy_v2_class",
) -> dict:
    rows = split_mod._read_manifest(str(manifest_csv))
    split_art = split_mod.build_split_artifact(
        rows, holdout_fraction=holdout_fraction, root=root
    )
    man_art = man_mod.build_versioned_manifest(
        rows,
        source=source,
        license_=license_,
        label_authority=label_authority,
        holdout_fraction=holdout_fraction,
        root=root,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = out_dir / "e1_split_artifact.json"
    man_path = out_dir / "e1_manifest_artifact.json"
    split_path.write_text(json.dumps(split_art, indent=2), encoding="utf-8")
    man_path.write_text(json.dumps(man_art, indent=2), encoding="utf-8")

    # Verify both artifacts (trusted caller holdout_fraction).
    split_mod.verify_reproducible(rows, split_art, root=root)
    man_mod.verify_manifest(
        rows,
        man_art,
        root=root,
        expected_holdout_fraction=holdout_fraction,
    )

    if split_art.get("unlocks_retraining") is not False:
        raise split_mod.IntegrityError("E1 artifact must hardcode unlocks_retraining=false")
    if man_art.get("unlocks_retraining") is not False:
        raise man_mod.IntegrityError("E1 manifest must hardcode unlocks_retraining=false")

    return {
        "split_artifact": str(split_path),
        "manifest_artifact": str(man_path),
        "split_digest": split_art.get("split_digest"),
        "manifest_digest": man_art.get("manifest_digest"),
        "eval_eligible": split_art.get("eval_eligible"),
        "unlocks_retraining": False,
        "schema_split": split_art.get("schema_version"),
        "schema_manifest": man_art.get("schema_version"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Track E E1 dry-run: split + versioned manifest + verify (torch-free)."
    )
    parser.add_argument("--manifest", required=True, help="Input CSV manifest path")
    parser.add_argument(
        "--root",
        default="",
        help="Dataset root for relative locators (optional; recommended for portability)",
    )
    parser.add_argument(
        "--out",
        default="artifacts/track_e_e1_dry_run",
        help="Output directory for dry-run artifacts",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=split_mod.DEFAULT_HOLDOUT_FRACTION,
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else None
    try:
        summary = run_dry_run(
            manifest_csv=Path(args.manifest),
            root=root,
            out_dir=Path(args.out),
            holdout_fraction=float(args.holdout_fraction),
        )
    except (split_mod.IntegrityError, man_mod.IntegrityError, OSError) as exc:
        sys.stderr.write(f"[track-e-e1] dry-run FAILED: {exc}\n")
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
