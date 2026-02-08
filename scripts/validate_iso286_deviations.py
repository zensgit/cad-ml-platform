#!/usr/bin/env python3
"""Validate ISO 286 / GB/T 1800 deviations table artifact.

This validates the JSON produced by `scripts/extract_iso286_deviations.py` and
used by `src/core/knowledge/tolerance`.

It is intentionally conservative:
- checks structure and monotonic size ranges,
- checks basic invariants (lower <= upper),
- optionally runs a few spot-check lookups through the public API helpers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_JSON_PATH = Path("data/knowledge/iso286_deviations.json")


@dataclass(frozen=True)
class ValidationIssue:
    severity: str  # "error" | "warning"
    message: str


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iter_rows(rows: Any) -> Iterable[Tuple[float, float, float]]:
    if not isinstance(rows, list):
        return
    for entry in rows:
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        size_upper = _as_float(entry[0])
        lower = _as_float(entry[1])
        upper = _as_float(entry[2])
        if size_upper is None or lower is None or upper is None:
            continue
        yield size_upper, lower, upper


def _validate_label_rows(
    kind: str,
    label: str,
    rows: Any,
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    cleaned = list(_iter_rows(rows))
    if not cleaned:
        issues.append(
            ValidationIssue(
                "error",
                f"{kind}.{label}: no valid rows (expect list of [size_upper, lower, upper])",
            )
        )
        return issues

    prev_size: Optional[float] = None
    for idx, (size_upper, lower, upper) in enumerate(cleaned):
        if size_upper <= 0:
            issues.append(
                ValidationIssue(
                    "error",
                    f"{kind}.{label}[{idx}]: size_upper must be > 0 (got {size_upper})",
                )
            )
        if prev_size is not None and size_upper <= prev_size:
            issues.append(
                ValidationIssue(
                    "error",
                    f"{kind}.{label}[{idx}]: size_upper not strictly increasing "
                    f"(prev={prev_size}, got={size_upper})",
                )
            )
        if lower > upper:
            issues.append(
                ValidationIssue(
                    "error",
                    f"{kind}.{label}[{idx}]: lower > upper (lower={lower}, upper={upper})",
                )
            )
        prev_size = size_upper
    return issues


def validate_json(path: Path) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    issues: List[ValidationIssue] = []
    if not path.exists():
        return [ValidationIssue("error", f"missing file: {path}")], {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [ValidationIssue("error", f"invalid json: {exc}")], {}

    if not isinstance(data, dict):
        return [ValidationIssue("error", "root must be a JSON object")], {}

    units = data.get("units")
    if units and str(units).strip().lower() != "um":
        issues.append(
            ValidationIssue("warning", f"unexpected units={units!r} (expected 'um')")
        )

    holes = data.get("holes")
    shafts = data.get("shafts")
    if not isinstance(holes, dict) or not isinstance(shafts, dict):
        issues.append(
            ValidationIssue(
                "error",
                "root must contain 'holes' and 'shafts' objects",
            )
        )
        return issues, data

    if not holes:
        issues.append(ValidationIssue("error", "holes table is empty"))
    if not shafts:
        issues.append(ValidationIssue("error", "shafts table is empty"))

    for kind, table in (("holes", holes), ("shafts", shafts)):
        for label, rows in table.items():
            if not isinstance(label, str) or not label.strip():
                issues.append(
                    ValidationIssue("error", f"{kind}: invalid label {label!r}")
                )
                continue
            issues.extend(_validate_label_rows(kind, label.strip(), rows))

    # Light sanity checks for common labels (avoid being overly strict).
    required_holes = ["H7", "JS6"]
    required_shafts = ["h6", "g6"]
    for key in required_holes:
        if key not in holes:
            issues.append(
                ValidationIssue("warning", f"holes missing common label: {key}")
            )
    for key in required_shafts:
        if key not in shafts:
            issues.append(
                ValidationIssue("warning", f"shafts missing common label: {key}")
            )

    return issues, data


def run_spot_checks(json_path: Path) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    # Ensure the knowledge module loads this exact file.
    os.environ["ISO286_DEVIATIONS_PATH"] = str(json_path)
    try:
        from src.core.knowledge.tolerance import (
            get_fit_deviations,
            get_limit_deviations,
        )
    except Exception as exc:  # noqa: BLE001
        return [ValidationIssue("error", f"failed to import tolerance module: {exc}")]

    # H7 @ 25mm -> (0, 21) um (see tests/integration/test_tolerance_api.py).
    h7 = get_limit_deviations("H", 7, 25.0)
    if h7 != (0.0, 21.0):
        issues.append(
            ValidationIssue("error", f"spot-check H7@25mm expected (0,21), got {h7}")
        )

    # g6 @ 10mm -> (-14, -5) um (see data/knowledge/iso286_deviations.json).
    g6 = get_limit_deviations("g", 6, 10.0)
    if g6 != (-14.0, -5.0):
        issues.append(
            ValidationIssue("error", f"spot-check g6@10mm expected (-14,-5), got {g6}")
        )

    fit = get_fit_deviations("H7/g6", 25.0)
    if fit is None:
        issues.append(
            ValidationIssue("error", "spot-check fit H7/g6@25mm returned None")
        )
    else:
        if (
            fit.hole_lower_deviation_um != 0.0
            or fit.hole_upper_deviation_um != 21.0
            or fit.shaft_upper_deviation_um != -7.0
            or fit.shaft_lower_deviation_um != -20.0
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "spot-check fit H7/g6@25mm deviations mismatch "
                    f"(hole={fit.hole_lower_deviation_um},{fit.hole_upper_deviation_um} "
                    f"shaft={fit.shaft_lower_deviation_um},{fit.shaft_upper_deviation_um})",
                )
            )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help="Path to iso286_deviations.json",
    )
    parser.add_argument(
        "--spot-check",
        action="store_true",
        help="Run a few deterministic spot-check lookups",
    )
    args = parser.parse_args()

    issues, data = validate_json(args.json)
    if args.spot_check:
        issues.extend(run_spot_checks(args.json))

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    holes_count = len(data.get("holes", {}) or {}) if isinstance(data, dict) else 0
    shafts_count = len(data.get("shafts", {}) or {}) if isinstance(data, dict) else 0
    print(
        f"ISO286 deviations: holes={holes_count} shafts={shafts_count} path={args.json}"
    )
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for item in warnings[:50]:
            print(f"  - {item.message}")
        if len(warnings) > 50:
            print(f"  ... truncated ({len(warnings) - 50} more)")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for item in errors[:50]:
            print(f"  - {item.message}")
        if len(errors) > 50:
            print(f"  ... truncated ({len(errors) - 50} more)")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
