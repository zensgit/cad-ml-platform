#!/usr/bin/env python3
"""Generate geometry rules from a DWG label manifest."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


_PAREN_PATTERN = re.compile(r"[()（）]")


def _slug(label: str) -> str:
    # Use a stable hash-like slug to avoid non-ASCII ids.
    import hashlib

    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:12]
    return f"label_{digest}"


def _split_variants(label: str) -> List[str]:
    variants = {label}
    cleaned = _PAREN_PATTERN.sub("", label)
    if cleaned and cleaned != label:
        variants.add(cleaned)
    if label.endswith("组件"):
        variants.add(label[: -len("组件")])
    if label.endswith("部件"):
        variants.add(label[: -len("部件")])
    if label.endswith("部分"):
        variants.add(label[: -len("部分")])
    return [v for v in variants if v]


def _load_manifest(path: Path) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    rows = path.read_text(encoding="utf-8").splitlines()
    if not rows:
        return labels
    header = rows[0].split(",")
    label_idx = header.index("label_cn")
    for line in rows[1:]:
        if not line.strip():
            continue
        parts = list(_parse_csv_line(line, len(header)))
        label = parts[label_idx].strip()
        if not label:
            continue
        labels[label] = labels.get(label, 0) + 1
    return labels


def _parse_csv_line(line: str, expected_cols: int) -> Iterable[str]:
    # Minimal CSV parsing for this manifest: split on commas, respecting quotes.
    import csv
    from io import StringIO

    reader = csv.reader(StringIO(line))
    row = next(reader)
    if len(row) < expected_cols:
        row.extend([""] * (expected_cols - len(row)))
    return row


def _load_synonyms(path: Path | None) -> Dict[str, List[str]]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return {str(k): list(v) for k, v in data.items() if isinstance(v, list)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate geometry rules from manifest.")
    parser.add_argument(
        "--manifest",
        default="reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv",
        help="Manifest CSV path",
    )
    parser.add_argument(
        "--rules-json",
        default="data/knowledge/geometry_rules.json",
        help="Geometry rules JSON path",
    )
    parser.add_argument(
        "--synonyms-json",
        default="",
        help="Optional JSON mapping label_cn -> [english synonyms]",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=90,
        help="Priority for generated rules",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(str(manifest_path))

    rules_path = Path(args.rules_json)
    if not rules_path.exists():
        raise FileNotFoundError(str(rules_path))

    synonyms = _load_synonyms(Path(args.synonyms_json)) if args.synonyms_json else {}

    payload = json.loads(rules_path.read_text(encoding="utf-8"))
    rules = payload.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError("Invalid rules JSON structure")

    existing_labels = set()
    rules_by_label: Dict[str, dict] = {}
    for rule in rules:
        part_hints = rule.get("part_hints") or {}
        for label in part_hints.keys():
            label_str = str(label)
            existing_labels.add(label_str)
            rules_by_label[label_str] = rule

    labels = _load_manifest(manifest_path)
    now = datetime.now(timezone.utc).isoformat()

    added = 0
    updated = 0
    if synonyms:
        for label, values in synonyms.items():
            rule = rules_by_label.get(label)
            if not rule or not values:
                continue
            keywords = rule.get("keywords") or []
            if not isinstance(keywords, list):
                keywords = []
            new_keywords = [str(v) for v in values if str(v).strip()]
            merged = list(dict.fromkeys([*keywords, *new_keywords]))
            if merged != keywords:
                rule["keywords"] = merged
                rule["updated_at"] = now
                metadata = rule.get("metadata") or {}
                if isinstance(metadata, dict):
                    metadata["synonyms_updated_at"] = now
                    rule["metadata"] = metadata
                updated += 1
    for label, count in sorted(labels.items(), key=lambda item: (-item[1], item[0])):
        if label in existing_labels:
            continue
        keywords = _split_variants(label)
        if label in synonyms:
            keywords.extend([str(s) for s in synonyms[label] if str(s).strip()])
        rule = {
            "id": f"dataset_{_slug(label)}",
            "category": "geometry",
            "name": f"Dataset rule for {label}",
            "chinese_name": label,
            "description": f"Auto-generated from manifest (count={count}).",
            "keywords": sorted(set(keywords)),
            "ocr_patterns": [],
            "part_hints": {label: 0.9},
            "enabled": True,
            "priority": int(args.priority),
            "source": "dataset_manifest",
            "created_at": now,
            "updated_at": now,
            "metadata": {"count": count},
        }
        rules.append(rule)
        added += 1

    payload["rules"] = rules
    payload["count"] = len(rules)
    payload["updated_at"] = now
    payload["version"] = now

    rules_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Added {added} rules, updated {updated} rules. Total rules: {len(rules)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
