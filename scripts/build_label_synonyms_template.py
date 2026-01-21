#!/usr/bin/env python3
"""Generate a template JSON file for English label synonyms."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _load_manifest(path: Path) -> List[str]:
    labels: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = (row.get("label_cn") or "").strip()
            if label and label not in labels:
                labels.append(label)
    return labels


def _load_existing(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    output: Dict[str, List[str]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            output[str(key)] = [str(item) for item in value if str(item).strip()]
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate label synonym template.")
    parser.add_argument(
        "--manifest",
        default="reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv",
        help="Manifest CSV path",
    )
    parser.add_argument(
        "--output-json",
        default="data/knowledge/label_synonyms_template.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--existing-json",
        default="",
        help="Optional existing JSON to merge",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(str(manifest_path))

    existing = (
        _load_existing(Path(args.existing_json)) if args.existing_json else {}
    )
    labels = _load_manifest(manifest_path)
    payload: Dict[str, List[str]] = {}
    for label in labels:
        payload[label] = existing.get(label, [])

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(payload)} label entries to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
