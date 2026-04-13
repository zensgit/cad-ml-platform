#!/usr/bin/env python3
"""Build Unified Manifest v2 — merges DXF file inventory with annotations
and taxonomy v2 mapping.

Scans data directories for DXF files, merges with annotation CSVs, applies
taxonomy v2 class mapping, and outputs a unified manifest with class
distribution and gap analysis.

Usage:
    python scripts/build_unified_manifest.py --dry-run
    python scripts/build_unified_manifest.py
    python scripts/build_unified_manifest.py --target-per-class 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "manifests" / "unified_manifest_v2.csv"
DEFAULT_SYNONYMS_PATH = ROOT / "data" / "knowledge" / "label_synonyms_template.json"
DEFAULT_TAXONOMY_PATH = ROOT / "config" / "label_taxonomy_v2.yaml"
DEFAULT_ANNOTATIONS_DIR = ROOT / "data" / "annotations"

# Data directories to scan for DXF files
SCAN_DIRS = [
    ROOT / "data" / "training",
    ROOT / "data" / "training_v2",
    ROOT / "data" / "training_v3",
    ROOT / "data" / "training_v5",
    ROOT / "data" / "training_v7",
    ROOT / "data" / "training_v8",
    ROOT / "data" / "training_4000",
    ROOT / "data" / "training_merged",
    ROOT / "data" / "training_merged_v2",
    ROOT / "data" / "standards_dxf",
    ROOT / "data" / "synthetic_v2",
    ROOT / "data" / "augmented",
]

# Reuse extraction logic from label_annotation_tool
_CN_CHAR_RE = re.compile(r"[\u4e00-\u9fa5()（）]+")
_VERSION_SUFFIX_RE = re.compile(r"(?:[_\-\s]?[vV]\d+)$")
_COMPARE_PREFIX_RE = re.compile(r"^比较[_\-\s]*")
_SPEC_SUFFIX_RES = (
    re.compile(r"(?:DN)\s*\d+(?:\.\d+)?$", re.IGNORECASE),
    re.compile(r"(?:PN)\s*\d+(?:\.\d+)?$", re.IGNORECASE),
    re.compile(r"(?:M)\s*\d+(?:x\d+(?:\.\d+)?)?$", re.IGNORECASE),
)


def extract_part_name(filename: str) -> Optional[str]:
    """Extract Chinese part name from filename."""
    if not filename:
        return None
    basename = Path(filename).stem
    if _COMPARE_PREFIX_RE.match(basename):
        basename = _COMPARE_PREFIX_RE.sub("", basename)
        if " vs " in basename.lower():
            basename = basename.lower().split(" vs ")[0].strip()
    basename = _VERSION_SUFFIX_RE.sub("", basename).strip()
    for _ in range(3):
        before = basename
        for spec_re in _SPEC_SUFFIX_RES:
            basename = spec_re.sub("", basename).strip()
        basename = re.sub(r"[_\-\s]+$", "", basename).strip()
        if basename == before:
            break
    cn_matches = _CN_CHAR_RE.findall(basename)
    if cn_matches:
        return max(cn_matches, key=len)
    return None


def load_synonyms(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if isinstance(v, list)}


def build_synonym_matcher(synonyms: Dict[str, List[str]]) -> Dict[str, str]:
    matcher: Dict[str, str] = {}
    for label, aliases in synonyms.items():
        matcher[label.lower()] = label
        for alias in aliases:
            if alias:
                matcher[alias.lower()] = label
    return matcher


def load_taxonomy_mapping(taxonomy_path: Path) -> Dict[str, str]:
    """Load taxonomy v2 old_label -> new_class mapping."""
    mapping: Dict[str, str] = {}
    if not taxonomy_path.exists():
        return mapping
    try:
        import yaml  # type: ignore[import-untyped]
        data = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
    except ImportError:
        return _parse_taxonomy_simple(taxonomy_path)
    if not data:
        return mapping
    for cls_name, cls_info in (data.get("classes") or {}).items():
        for src_label in (cls_info.get("source_labels") or []):
            mapping[src_label] = cls_name
    for src_label, target in (data.get("special_mappings") or {}).items():
        mapping[src_label] = target
    return mapping


def _parse_taxonomy_simple(path: Path) -> Dict[str, str]:
    """Regex fallback for parsing taxonomy YAML without PyYAML."""
    mapping: Dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    current_class = None
    in_source_labels = False
    in_special = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("special_mappings:"):
            in_special = True
            in_source_labels = False
            current_class = None
            continue
        if in_special:
            m = re.match(r'"(.+?)"\s*:\s*"(.+?)"', stripped)
            if m:
                mapping[m.group(1)] = m.group(2)
            elif stripped and not stripped.startswith("#") and ":" in stripped and not stripped.startswith('"'):
                in_special = False
            continue
        if re.match(r"^  \S", line) and not stripped.startswith("-") and not stripped.startswith("#"):
            m = re.match(r"^  (\S+)\s*:", line)
            if m:
                candidate = m.group(1)
                if candidate not in ("id", "description", "source_labels", "synonyms"):
                    current_class = candidate
                    in_source_labels = False
        if stripped == "source_labels:":
            in_source_labels = True
            continue
        if in_source_labels:
            if stripped.startswith("- "):
                label = stripped[2:].strip().strip('"').strip("'")
                if current_class and label:
                    mapping[label] = current_class
            else:
                in_source_labels = False
    return mapping


def load_excluded_labels(taxonomy_path: Path) -> Set[str]:
    """Load the excluded_labels list from taxonomy."""
    excluded: Set[str] = set()
    if not taxonomy_path.exists():
        return excluded
    try:
        import yaml  # type: ignore[import-untyped]
        data = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
        for label in (data.get("excluded_labels") or []):
            excluded.add(label)
    except ImportError:
        text = taxonomy_path.read_text(encoding="utf-8")
        in_excluded = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped == "excluded_labels:":
                in_excluded = True
                continue
            if in_excluded:
                if stripped.startswith("- "):
                    excluded.add(stripped[2:].strip().strip('"').strip("'"))
                else:
                    in_excluded = False
    return excluded


def load_annotations(annotations_dir: Path) -> Dict[str, dict]:
    """Load all annotation CSVs and return file_path -> annotation dict."""
    annotations: Dict[str, dict] = {}
    if not annotations_dir.exists():
        return annotations
    for csv_path in sorted(annotations_dir.glob("*.csv")):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fp = row.get("file_path", "")
                if fp:
                    annotations[fp] = row
    return annotations


def iter_dxf_files(dirs: List[Path]) -> List[Path]:
    """Scan multiple directories for DXF files."""
    seen: Set[str] = set()
    result: List[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        for pattern in ["*.dxf", "*.DXF"]:
            for p in sorted(d.rglob(pattern)):
                key = str(p).lower()
                if key not in seen:
                    seen.add(key)
                    result.append(p)
    return result


def match_synonym(part_name: str, matcher: Dict[str, str]) -> Optional[str]:
    key = part_name.lower().strip()
    if key in matcher:
        return matcher[key]
    best, best_len = None, 0
    for mk, label in matcher.items():
        if mk in key or key in mk:
            overlap = min(len(mk), len(key))
            if overlap > best_len and overlap >= 2:
                best_len = overlap
                best = label
    return best


def run_build(
    output_path: Path,
    synonyms_path: Path,
    taxonomy_path: Path,
    annotations_dir: Path,
    target_per_class: int = 30,
    dry_run: bool = False,
) -> None:
    """Build the unified manifest."""
    synonyms = load_synonyms(synonyms_path)
    matcher = build_synonym_matcher(synonyms)
    taxonomy_map = load_taxonomy_mapping(taxonomy_path)
    excluded = load_excluded_labels(taxonomy_path)
    annotations = load_annotations(annotations_dir)

    dxf_files = iter_dxf_files(SCAN_DIRS)
    print(f"Scanned {len(SCAN_DIRS)} directories, found {len(dxf_files)} DXF files")
    print(f"Loaded {len(annotations)} annotations")
    if dry_run:
        print("[DRY RUN] No files will be written.\n")

    fieldnames = [
        "file_path", "filename", "extracted_name", "synonym_label",
        "taxonomy_v2_class", "source", "split", "timestamp",
    ]

    rows: List[dict] = []
    class_counter: Counter = Counter()
    unmapped_counter: Counter = Counter()
    excluded_count = 0

    for dxf_path in dxf_files:
        fpath_str = str(dxf_path)
        part_name = extract_part_name(dxf_path.name)

        # Check if we have an annotation override
        ann = annotations.get(fpath_str)
        if ann and ann.get("annotator_label"):
            annotator_label = ann["annotator_label"]
            # Try to map annotator label to taxonomy
            tax_class = taxonomy_map.get(annotator_label, annotator_label)
        else:
            syn_label = match_synonym(part_name, matcher) if part_name else None
            if syn_label and syn_label in excluded:
                excluded_count += 1
                continue
            tax_class = taxonomy_map.get(syn_label, "") if syn_label else ""
            annotator_label = ""

        # Also check parent directory for by_class datasets
        parent_name = dxf_path.parent.name
        if not tax_class and parent_name in taxonomy_map:
            tax_class = taxonomy_map[parent_name]
        if not tax_class and parent_name == "by_class":
            # grandparent might be the class
            grandparent = dxf_path.parent.parent.name
            if grandparent in taxonomy_map:
                tax_class = taxonomy_map[grandparent]

        syn_label_display = match_synonym(part_name, matcher) if part_name else ""

        if not tax_class:
            if part_name:
                unmapped_counter[part_name] += 1
            continue

        # Determine source
        source = "annotation" if ann else "auto"
        # Determine split placeholder
        split = ""

        row = {
            "file_path": fpath_str,
            "filename": dxf_path.name,
            "extracted_name": part_name or "",
            "synonym_label": syn_label_display or "",
            "taxonomy_v2_class": tax_class,
            "source": source,
            "split": split,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(row)
        class_counter[tax_class] += 1

    # Print class distribution
    print(f"\n{'='*60}")
    print(f"Class Distribution (taxonomy v2) — {len(rows)} labeled samples")
    print(f"{'='*60}")
    print(f"{'Class':<16} {'Count':>6}  {'Gap':>6}  Status")
    print(f"{'-'*16} {'-'*6}  {'-'*6}  {'-'*10}")

    total_gap = 0
    for cls, count in sorted(class_counter.items(), key=lambda x: -x[1]):
        gap = max(0, target_per_class - count)
        total_gap += gap
        status = "OK" if gap == 0 else f"NEED +{gap}"
        print(f"{cls:<16} {count:>6}  {gap:>6}  {status}")

    print(f"\nTotal classes: {len(class_counter)}")
    print(f"Total labeled samples: {len(rows)}")
    print(f"Excluded (noise/educational): {excluded_count}")
    print(f"Unmapped (no taxonomy class): {sum(unmapped_counter.values())}")
    print(f"Total data gap (target={target_per_class}/class): {total_gap} samples needed")

    if unmapped_counter:
        print(f"\nTop 15 unmapped part names:")
        for name, count in unmapped_counter.most_common(15):
            print(f"  {name}: {count}")

    if dry_run:
        return

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nManifest written to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified manifest v2 with taxonomy mapping")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--synonyms", type=Path, default=DEFAULT_SYNONYMS_PATH)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--target-per-class", type=int, default=30, help="Target samples per class for gap analysis")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    run_build(
        output_path=args.output,
        synonyms_path=args.synonyms,
        taxonomy_path=args.taxonomy,
        annotations_dir=args.annotations_dir,
        target_per_class=args.target_per_class,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
