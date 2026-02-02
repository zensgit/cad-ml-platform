#!/usr/bin/env python3
"""Build ISO 286 hole deviation table.

Sources:
- isofits PyPI data (if installed)
- Derived from shaft fundamental deviations for missing symbols (C/D)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


OUTPUT_PATH = Path("data/knowledge/iso286_hole_deviations.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_entry(entry: str) -> Tuple[float, float] | None:
    nums = [float(x) for x in re.findall(r"[+-]?\d+\.?\d*", entry)]
    if len(nums) < 2:
        return None
    return nums[0], nums[1]


def _load_isofits() -> tuple[Dict[str, List[List[float]]], List[float]]:
    try:
        import importlib.util
        import isofits

        data_path = Path(isofits.__file__).resolve().with_name("data.py")
        if not data_path.exists():
            return {}, []
        spec = importlib.util.spec_from_file_location("isofits_data", data_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader
        spec.loader.exec_module(mod)
        hole_data = getattr(mod, "hole_data", {})
        size_uppers = [float(x) for x in hole_data.get("inc.", [])]
        symbol_grade = {
            "E": "E6",
            "F": "F6",
            "G": "G6",
            "H": "H6",
            "J": "J6",
            "JS": "JS6",
            "K": "K6",
            "M": "M6",
            "N": "N6",
            "P": "P6",
            "R": "R6",
        }
        deviations: Dict[str, List[List[float]]] = {}
        for symbol, key in symbol_grade.items():
            entries = hole_data.get(key)
            if not entries or len(entries) != len(size_uppers):
                continue
            devs: List[List[float]] = []
            for size_upper, entry in zip(size_uppers, entries):
                parsed = _parse_entry(entry)
                if parsed is None:
                    continue
                _, lower = parsed
                devs.append([size_upper, lower])
            if devs:
                deviations[symbol] = devs
        return deviations, size_uppers
    except ImportError:
        return {}, []
    except Exception:
        return {}, []


def _derive_from_shafts() -> Dict[str, List[List[float]]]:
    try:
        from src.core.knowledge.tolerance.fits import SHAFT_FUNDAMENTAL_DEVIATIONS
    except Exception:
        return {}

    derived: Dict[str, List[List[float]]] = {}
    for symbol in ("c", "d"):
        table = SHAFT_FUNDAMENTAL_DEVIATIONS.get(symbol)
        if not table:
            continue
        devs: List[List[float]] = []
        for size_upper, upper_dev in table:
            # For corresponding hole symbol, EI ~= -ES(shaft)
            devs.append([float(size_upper), float(-upper_dev)])
        derived[symbol.upper()] = devs
    return derived


def build() -> dict:
    deviations, _ = _load_isofits()
    derived = _derive_from_shafts()
    for symbol, table in derived.items():
        deviations.setdefault(symbol, table)
    return {
        "source": "ISO 286-2 (isofits data + derived from shaft deviations)",
        "units": "um",
        "notes": "Lower deviation (EI) per symbol and size upper bound. C/D derived from shaft deviations when ISO table is missing.",
        "deviations": deviations,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build()
    if not payload.get("deviations"):
        raise SystemExit(
            "ISO 286 deviations not available. Install 'isofits' and rerun."
        )
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    symbols = ", ".join(sorted(payload.get("deviations", {}).keys()))
    print(f"Wrote {OUTPUT_PATH} with symbols: {symbols}")


if __name__ == "__main__":
    main()
