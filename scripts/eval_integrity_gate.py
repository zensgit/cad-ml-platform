#!/usr/bin/env python3
"""Fail-closed evaluation-integrity gate (L3).

`scripts/auto_retrain.sh` calls this BEFORE any manifest write, preprocessing, or training.
Re-enablement of the retraining / model-promotion path is bound to a **versioned, reproducible
evaluation-integrity artifact** (PRODUCT_STRATEGY.md §5.2 model-promotion gate, §8.1 Track E:
evaluation-integrity-v2) — never to queue-row counts or an environment toggle. A missing,
invalid, or version-mismatched artifact makes this exit non-zero, so the pipeline stops before
mutating anything.

Scope (L3, deliberately minimal): this validates that a *conforming, versioned* artifact is
present. It does NOT re-run the evaluation or re-verify reproducibility from source — producing a
genuinely reproducible artifact (content-hash + normalized-family split, holdout, versioned result
artifact) is Track E's job (§8.1). L3 establishes the contract Track E must satisfy; it does not
defend against a hand-forged artifact (§8 forbids an env-toggle bypass, not forgery).

Stdlib only, so the fail-closed path runs without the ML stack installed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_VERSION = "evaluation-integrity-v2"

# Fields a real Track E (§8.1) artifact carries. Presence + non-emptiness is the L3 contract;
# `reproducible` must be explicitly true (a placeholder that hasn't been reproduced is red).
REQUIRED_FIELDS = (
    "schema_version",
    "split_strategy",   # e.g. content-hash+normalized-family (§8.1.1) — NOT path-only
    "holdout",          # customer-family or time-based holdout (§8.1.3)
    "metrics",          # per-class/macro/calibration/false-duplicate/missed-reuse (§8.1.4)
    "label_authority",  # provenance / label authority (§8.1.6)
    "reproducible",     # exit condition: a fresh clone can reproduce the result (§8.1)
)

# §8.1.1: the split must be content-hash + normalized-family, NOT a path-only check. The gate owns
# this canonical value; a Track E artifact must match it exactly (a wrong/typo/type-confused value
# — including a bare `true` — is red, not silently accepted).
REQUIRED_SPLIT_STRATEGY = "content-hash+normalized-family"

# §8.1.4: the metric families a real evaluation reports. All must be present in `metrics`.
REQUIRED_METRIC_KEYS = (
    "per_class",
    "macro_f1",
    "calibration_ece",
    "false_duplicate_rate",
    "missed_reuse_rate",
)

_STRATEGY_REF = (
    "See docs/PRODUCT_STRATEGY.md §5.2 (model-promotion gate) and "
    "§8.1 (Track E: evaluation-integrity-v2)."
)


class GateError(Exception):
    """A fail-closed reason. `reason` is one of: missing / invalid / version-mismatch."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


def validate_artifact(path: str, *, require_version: str = REQUIRED_VERSION) -> dict:
    """Return the parsed artifact if it is a valid, version-matched evaluation-integrity artifact.

    Raises GateError(kind, detail) for the three fail-closed modes. Never returns on failure.
    """
    p = Path(path)
    if not p.is_file():
        raise GateError("missing", f"no evaluation-integrity artifact at {path!r}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise GateError("invalid", f"artifact {path!r} is not readable JSON: {exc}")

    if not isinstance(data, dict):
        raise GateError("invalid", f"artifact {path!r} must be a JSON object")

    version = data.get("schema_version")
    if not version:
        raise GateError("invalid", f"artifact {path!r} has no schema_version")
    if version != require_version:
        raise GateError(
            "version-mismatch",
            f"artifact schema_version {version!r} != required {require_version!r}",
        )

    absent = [f for f in REQUIRED_FIELDS if f not in data]
    if absent:
        raise GateError("invalid", f"artifact {path!r} missing required fields: {', '.join(absent)}")

    # Per-field TYPE + VALUE checks. Presence alone is not enough: a schema bug in the Track E
    # producer (or a forged artifact) can set a field to the wrong type — e.g. `true` — and
    # non-empty-only validation would wrongly open the training path.
    if data["split_strategy"] != REQUIRED_SPLIT_STRATEGY:
        raise GateError(
            "invalid",
            f"split_strategy must equal {REQUIRED_SPLIT_STRATEGY!r} (§8.1.1: content-hash + "
            f"normalized-family, not path-only), got {data['split_strategy']!r}",
        )

    if not isinstance(data["holdout"], dict) or not data["holdout"]:
        raise GateError("invalid", "holdout must be a non-empty object (§8.1.3)")

    metrics = data["metrics"]
    if not isinstance(metrics, dict):
        raise GateError("invalid", "metrics must be an object (§8.1.4)")
    missing_metrics = [k for k in REQUIRED_METRIC_KEYS if k not in metrics]
    if missing_metrics:
        raise GateError(
            "invalid", f"metrics missing required families (§8.1.4): {', '.join(missing_metrics)}"
        )

    label_authority = data["label_authority"]
    if not (
        (isinstance(label_authority, str) and label_authority.strip())
        or (isinstance(label_authority, dict) and label_authority)
    ):
        raise GateError(
            "invalid", "label_authority must be a non-empty string or a non-empty object (§8.1.6)"
        )

    # Strict identity: a truthy string/number ("true", 1) does NOT satisfy the reproducible gate.
    if data["reproducible"] is not True:
        raise GateError(
            "invalid",
            f"artifact {path!r} does not assert reproducible=true "
            "(evaluation integrity not established)",
        )

    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed evaluation-integrity gate for the retraining pipeline."
    )
    parser.add_argument(
        "--artifact",
        required=True,
        help="path to the evaluation-integrity-v2 artifact (JSON)",
    )
    parser.add_argument(
        "--require-version",
        default=REQUIRED_VERSION,
        help=f"required schema_version (default: {REQUIRED_VERSION})",
    )
    args = parser.parse_args(argv)

    try:
        validate_artifact(args.artifact, require_version=args.require_version)
    except GateError as exc:
        sys.stderr.write(
            "EVALUATION-INTEGRITY GATE (fail-closed): "
            f"{exc.kind} — {exc.detail}\n"
            "Retraining and model promotion are blocked until a valid, versioned "
            f"{args.require_version} artifact exists.\n"
            f"{_STRATEGY_REF}\n"
            "This gate is not disableable by an environment toggle (per §8).\n"
        )
        return 1

    sys.stdout.write(
        f"evaluation-integrity gate PASS: {args.artifact} ({args.require_version})\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
