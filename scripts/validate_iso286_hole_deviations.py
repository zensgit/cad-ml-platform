#!/usr/bin/env python3
"""Validate ISO 286 hole deviation coverage for COMMON_FITS usage."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Set

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.knowledge.tolerance.fits import COMMON_FITS


def main() -> None:
    path = Path("data/knowledge/iso286_hole_deviations.json")
    if not path.exists():
        raise SystemExit("iso286_hole_deviations.json not found")

    data = json.loads(path.read_text(encoding="utf-8"))
    deviations = data.get("deviations", {})
    symbols = {str(k).upper() for k in deviations.keys()}

    required: Set[str] = set()
    for fit in COMMON_FITS.values():
        hole_symbol = str(fit["hole"][0]).upper()
        if hole_symbol not in {"H", "JS"}:
            required.add(hole_symbol)

    missing = sorted(required - symbols)
    if missing:
        print("Missing hole deviation symbols:", ", ".join(missing))
        raise SystemExit(1)

    print("All required hole symbols present:", ", ".join(sorted(required)) or "None")


if __name__ == "__main__":
    main()
