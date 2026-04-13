#!/usr/bin/env python3
"""Label Annotation Tool — CLI for annotating DXF files with taxonomy v2 labels.

Usage:
    python scripts/label_annotation_tool.py --input-dir data/training/ --dry-run
    python scripts/label_annotation_tool.py --input-dir data/training/

For each DXF file: extracts Chinese part name from filename, matches against
label_synonyms_template.json, maps to taxonomy v2 class.  Interactive mode lets
the user accept [Enter], type a new label, [s]kip, or [q]uit.

Outputs to data/annotations/manifest_annotated.csv (configurable).
Supports --dry-run and resume (skips already-annotated files).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Filename extraction regex (self-contained, no imports from src/)
# ---------------------------------------------------------------------------
_CN_CHAR_RE = re.compile(r"[\u4e00-\u9fa5()（）]+")
_VERSION_SUFFIX_RE = re.compile(r"(?:[_\-\s]?[vV]\d+)$")
_COMPARE_PREFIX_RE = re.compile(r"^比较[_\-\s]*")
_SPEC_SUFFIX_RES = (
    re.compile(r"(?:DN)\s*\d+(?:\.\d+)?$", re.IGNORECASE),
    re.compile(r"(?:PN)\s*\d+(?:\.\d+)?$", re.IGNORECASE),
    re.compile(r"(?:M)\s*\d+(?:x\d+(?:\.\d+)?)?$", re.IGNORECASE),
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNONYMS_PATH = ROOT / "data" / "knowledge" / "label_synonyms_template.json"
DEFAULT_TAXONOMY_PATH = ROOT / "config" / "label_taxonomy_v2.yaml"
DEFAULT_OUTPUT = ROOT / "data" / "annotations" / "manifest_annotated.csv"


def extract_part_name(filename: str) -> Optional[str]:
    """Extract Chinese part name from a DXF filename."""
    if not filename:
        return None
    basename = Path(filename).stem
    if _COMPARE_PREFIX_RE.match(basename):
        basename = _COMPARE_PREFIX_RE.sub("", basename)
        if " vs " in basename.lower():
            basename = basename.lower().split(" vs ")[0].strip()
    basename = _VERSION_SUFFIX_RE.sub("", basename).strip()

    # Strip spec suffixes
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
    """Load the label_synonyms_template.json."""
    if not path.exists():
        print(f"WARNING: synonyms file not found: {path}", file=sys.stderr)
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if isinstance(v, list)}


def build_synonym_matcher(synonyms: Dict[str, List[str]]) -> Dict[str, str]:
    """Build lowered key -> canonical label mapping."""
    matcher: Dict[str, str] = {}
    for label, aliases in synonyms.items():
        matcher[label.lower()] = label
        for alias in aliases:
            if alias:
                matcher[alias.lower()] = label
    return matcher


def load_taxonomy_mapping(taxonomy_path: Path) -> Dict[str, str]:
    """Build old_label -> new_class mapping from taxonomy v2 YAML.

    Uses a simple parser to avoid requiring PyYAML at runtime.
    """
    mapping: Dict[str, str] = {}
    if not taxonomy_path.exists():
        print(f"WARNING: taxonomy file not found: {taxonomy_path}", file=sys.stderr)
        return mapping
    try:
        import yaml  # type: ignore[import-untyped]
        data = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
    except ImportError:
        # Fallback: parse source_labels manually
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
    """Regex-based fallback taxonomy parser when PyYAML is unavailable."""
    mapping: Dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    current_class = None
    in_source_labels = False
    in_special = False

    for line in text.splitlines():
        stripped = line.strip()

        # Detect special_mappings section
        if stripped.startswith("special_mappings:"):
            in_special = True
            in_source_labels = False
            current_class = None
            continue
        if in_special:
            m = re.match(r'"(.+?)"\s*:\s*"(.+?)"', stripped)
            if m:
                mapping[m.group(1)] = m.group(2)
            elif stripped and not stripped.startswith("#") and ":" in stripped and not stripped.startswith("-"):
                # End of special_mappings
                if not stripped.startswith('"'):
                    in_special = False
            continue

        # Detect class name (top-level key under classes)
        if re.match(r"^  \S", line) and not line.strip().startswith("-") and not line.strip().startswith("#"):
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


def match_to_synonym(part_name: str, matcher: Dict[str, str]) -> Optional[str]:
    """Match a part name to the synonym table (exact then partial)."""
    key = part_name.lower().strip()
    if key in matcher:
        return matcher[key]
    # Partial: longest overlap
    best, best_len = None, 0
    for mk, label in matcher.items():
        if mk in key or key in mk:
            overlap = min(len(mk), len(key))
            if overlap > best_len and overlap >= 2:
                best_len = overlap
                best = label
    return best


def load_already_annotated(output_path: Path) -> set:
    """Load set of already-annotated file paths for resume."""
    done: set = set()
    if not output_path.exists():
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "file_path" in row:
                done.add(row["file_path"])
    return done


def iter_dxf_files(input_dir: Path, recursive: bool = True) -> List[Path]:
    """Find all DXF files under input_dir."""
    patterns = ["*.dxf", "*.DXF"]
    paths: List[Path] = []
    for pattern in patterns:
        if recursive:
            paths.extend(sorted(input_dir.rglob(pattern)))
        else:
            paths.extend(sorted(input_dir.glob(pattern)))
    seen: set = set()
    unique: List[Path] = []
    for p in paths:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def run_annotation(
    input_dir: Path,
    output_path: Path,
    synonyms_path: Path,
    taxonomy_path: Path,
    dry_run: bool = False,
    non_interactive: bool = False,
    recursive: bool = True,
) -> None:
    """Main annotation loop."""
    synonyms = load_synonyms(synonyms_path)
    matcher = build_synonym_matcher(synonyms)
    taxonomy_map = load_taxonomy_mapping(taxonomy_path)

    dxf_files = iter_dxf_files(input_dir, recursive=recursive)
    already_done = load_already_annotated(output_path) if not dry_run else set()

    print(f"Found {len(dxf_files)} DXF files in {input_dir}")
    print(f"Already annotated: {len(already_done)}")
    if dry_run:
        print("[DRY RUN] No files will be written.\n")

    # Ensure output directory exists
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open CSV for appending
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    csv_file = None
    writer = None
    fieldnames = [
        "file_path", "filename", "extracted_name", "synonym_label",
        "taxonomy_v2_class", "annotator_label", "timestamp",
    ]

    if not dry_run:
        csv_file = open(output_path, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

    stats: Dict[str, int] = {"total": 0, "skipped": 0, "annotated": 0, "auto": 0}

    try:
        for dxf_path in dxf_files:
            fpath_str = str(dxf_path)
            if fpath_str in already_done:
                stats["skipped"] += 1
                continue

            stats["total"] += 1
            part_name = extract_part_name(dxf_path.name)
            syn_label = match_to_synonym(part_name, matcher) if part_name else None
            tax_class = taxonomy_map.get(syn_label, "") if syn_label else ""

            suggestion = tax_class or syn_label or part_name or "?"

            if dry_run:
                print(f"  {dxf_path.name}")
                print(f"    extracted: {part_name}  synonym: {syn_label}  class: {tax_class}")
                continue

            if non_interactive:
                annotator_label = suggestion if suggestion != "?" else ""
                stats["auto"] += 1
            else:
                print(f"\n[{stats['total']}] {dxf_path.name}")
                print(f"  extracted: {part_name}  synonym: {syn_label}  taxonomy: {tax_class}")
                prompt_text = f"  Label [{suggestion}] (Enter=accept, type new, s=skip, q=quit): "
                try:
                    user_input = input(prompt_text).strip()
                except EOFError:
                    break
                if user_input.lower() == "q":
                    break
                if user_input.lower() == "s":
                    stats["skipped"] += 1
                    continue
                annotator_label = user_input if user_input else suggestion

            row = {
                "file_path": fpath_str,
                "filename": dxf_path.name,
                "extracted_name": part_name or "",
                "synonym_label": syn_label or "",
                "taxonomy_v2_class": tax_class,
                "annotator_label": annotator_label,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if writer:
                writer.writerow(row)
            stats["annotated"] += 1

    finally:
        if csv_file:
            csv_file.close()

    print(f"\nDone. total={stats['total']} annotated={stats['annotated']} "
          f"skipped={stats['skipped']} auto={stats['auto']}")
    if not dry_run:
        print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate DXF files with taxonomy v2 labels",
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory of DXF files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--synonyms", type=Path, default=DEFAULT_SYNONYMS_PATH, help="Synonyms JSON path")
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH, help="Taxonomy v2 YAML path")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--non-interactive", action="store_true", help="Auto-accept suggestions")
    parser.add_argument("--no-recursive", action="store_true", help="Do not recurse into subdirectories")
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"ERROR: input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    run_annotation(
        input_dir=args.input_dir,
        output_path=args.output,
        synonyms_path=args.synonyms,
        taxonomy_path=args.taxonomy,
        dry_run=args.dry_run,
        non_interactive=args.non_interactive,
        recursive=not args.no_recursive,
    )


if __name__ == "__main__":
    main()
