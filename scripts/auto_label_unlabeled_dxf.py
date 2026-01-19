#!/usr/bin/env python3
"""Auto-label unlabeled DWG entries using DXF text hints and knowledge rules."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import ezdxf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.knowledge.dynamic.manager import get_knowledge_manager

logger = logging.getLogger(__name__)


def _load_synonyms(path: Path | None) -> Dict[str, List[str]]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    output: Dict[str, List[str]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            output[str(key)] = [str(item) for item in value if str(item).strip()]
    return output


def _extract_text(doc: ezdxf.EzdxfDocument) -> str:
    texts: List[str] = []
    for entity in doc.modelspace():
        dtype = entity.dxftype()
        if dtype == "TEXT":
            text = getattr(entity.dxf, "text", "")
            if text:
                texts.append(str(text))
        elif dtype == "MTEXT":
            text = getattr(entity, "text", "")
            if text:
                texts.append(str(text))
    return " ".join(t.strip() for t in texts if t.strip())


def _pick_label(text: str) -> Tuple[str, float]:
    if not text:
        return "", 0.0
    km = get_knowledge_manager()
    hints = km.get_part_hints(text=text, geometric_features=None, entity_counts=None)
    if not hints:
        return "", 0.0
    label, score = max(hints.items(), key=lambda item: item[1])
    return str(label), float(score)


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-label unlabeled DWG entries.")
    parser.add_argument(
        "--input-csv",
        default="reports/MECH_4000_DWG_UNLABELED_LABELS_TEMPLATE_20260119.csv",
        help="Unlabeled mapping CSV",
    )
    parser.add_argument(
        "--dxf-dir",
        required=True,
        help="DXF directory containing converted files",
    )
    parser.add_argument(
        "--synonyms-json",
        default="data/knowledge/label_synonyms_template.json",
        help="Synonyms JSON (for English labels)",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Output CSV (defaults to overwrite input)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    dxf_dir = Path(args.dxf_dir)
    if not dxf_dir.exists():
        raise FileNotFoundError(str(dxf_dir))

    synonyms = _load_synonyms(Path(args.synonyms_json))

    rows: List[Dict[str, str]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))

    for row in rows:
        file_name = row.get("file_name", "")
        stem = Path(file_name).stem
        dxf_path = dxf_dir / f"{stem}.dxf"
        label = ""
        label_en = ""
        score = 0.0
        note = "auto_label:missing_dxf"
        if dxf_path.exists():
            try:
                doc = ezdxf.readfile(dxf_path)
                text = _extract_text(doc)
                label, score = _pick_label(text)
                if label:
                    candidates = synonyms.get(label, [])
                    label_en = candidates[0] if candidates else ""
                    note = f"auto_label:text_hint score={score:.2f}"
                else:
                    note = "auto_label:no_text_match"
            except Exception as exc:
                logger.warning("DXF parse failed for %s: %s", dxf_path, exc)
                note = f"auto_label:parse_error {exc}"
        row["label_cn"] = label
        row["label_en"] = label_en
        row["notes"] = note

    output_path = Path(args.output_csv) if args.output_csv else input_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_name", "relative_path", "source_dir", "label_cn", "label_en", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote auto-labeled rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
