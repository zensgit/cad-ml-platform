#!/usr/bin/env python3
"""Generate a manual review sheet from DXF files.

Extracts text content, title block fields, and dimension counts to support
manual labeling/verification. Optionally renders preview PNGs for a subset.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import ezdxf  # type: ignore

from src.core.dedupcad_precision.cad_pipeline import (
    DxfRenderConfig,
    extract_geom_json_from_dxf,
    render_dxf_to_png,
)
from src.core.dedupcad_precision.vendor.parsers import parse_text_content
from src.core.ocr.parsing.title_block_parser import parse_title_block


def _load_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = (row.get("file_name") or "").strip()
            if not file_name:
                continue
            stem = Path(file_name).stem
            if stem not in mapping:
                mapping[stem] = {k: str(v) for k, v in row.items() if v is not None}
    return mapping


def _iter_layouts(doc: ezdxf.EzdxfDocument) -> Iterable[Any]:
    yield doc.modelspace()
    for layout in doc.layouts:
        if layout.name.lower() == "model":
            continue
        yield layout


def _extract_raw_text(doc: ezdxf.EzdxfDocument, max_items: int = 5000) -> List[str]:
    raw_items: List[str] = []
    budget = max(0, int(max_items))
    for layout in _iter_layouts(doc):
        for entity in layout:
            if len(raw_items) >= budget:
                return raw_items
            try:
                etype = entity.dxftype()
            except Exception:
                continue
            if etype not in {"TEXT", "MTEXT"}:
                continue
            text = ""
            if etype == "MTEXT":
                try:
                    if hasattr(entity, "plain_text"):
                        text = entity.plain_text()  # type: ignore[call-arg]
                    else:
                        text = getattr(entity, "text", "") or getattr(entity.dxf, "text", "")
                except Exception:
                    text = getattr(entity, "text", "") or getattr(entity.dxf, "text", "")
            else:
                try:
                    text = getattr(entity.dxf, "text", "") or ""
                except Exception:
                    text = ""
            text = str(text or "").replace("\n", " ").replace("\r", " ").strip()
            if text:
                raw_items.append(text)
    return raw_items


def _sample_dimensions(dimensions: List[Dict[str, Any]], limit: int = 5) -> str:
    sample = []
    for dim in dimensions[:limit]:
        dtype = dim.get("dimension_type")
        text = dim.get("measurement_text")
        if text:
            sample.append(f"{dtype}:{text}")
        elif dtype:
            sample.append(str(dtype))
    return " | ".join(sample)


def _truncate(values: List[str], max_chars: int = 200) -> str:
    text = " | ".join([v for v in values if v])
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _normalize_texts(values: List[str]) -> List[str]:
    normalized = []
    for value in values:
        cleaned = str(value or "").strip().lower()
        if cleaned:
            normalized.append(cleaned)
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate DXF manual review sheet.")
    parser.add_argument(
        "--dxf-dir",
        required=True,
        help="Directory containing DXF files.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional manifest CSV for labels/metadata.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output review CSV path.",
    )
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--preview-dir", default="")
    parser.add_argument("--preview-count", type=int, default=0)
    parser.add_argument("--render-dpi", type=int, default=200)
    parser.add_argument("--render-size", type=int, default=1024)
    args = parser.parse_args()

    dxf_dir = Path(args.dxf_dir)
    if not dxf_dir.exists():
        raise FileNotFoundError(str(dxf_dir))

    manifest = _load_manifest(Path(args.manifest)) if args.manifest else {}

    dxf_paths = sorted(
        [*dxf_dir.glob("*.dxf"), *dxf_dir.glob("*.DXF")],
        key=lambda p: p.name,
    )
    if not dxf_paths:
        raise SystemExit("No DXF files found.")

    random.seed(args.seed)
    sample_size = min(max(1, args.sample_size), len(dxf_paths))
    sample_paths = random.sample(dxf_paths, sample_size)

    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    preview_limit = max(0, int(args.preview_count))
    preview_paths = set(sample_paths[:preview_limit]) if preview_dir else set()
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file_name",
        "source_path",
        "relative_path",
        "source_dir",
        "suggested_label_cn",
        "suggested_label_en",
        "text_sample",
        "normalized_text_sample",
        "title_block_drawing_number",
        "title_block_part_name",
        "title_block_material",
        "title_block_scale",
        "title_block_sheet",
        "title_block_date",
        "title_block_weight",
        "title_block_company",
        "title_block_projection",
        "title_block_revision",
        "review_drawing_number",
        "review_part_name",
        "review_material",
        "review_scale",
        "review_sheet",
        "review_date",
        "review_weight",
        "review_company",
        "review_projection",
        "review_revision",
        "dimensions_count",
        "dimensions_sample",
        "layers_count",
        "hatches_count",
        "reviewer_label_cn",
        "reviewer_label_en",
        "review_status",
        "review_notes",
        "preview_path",
    ]

    render_cfg = DxfRenderConfig(size_px=args.render_size, dpi=args.render_dpi)

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dxf_path in sample_paths:
            stem = dxf_path.stem
            meta = manifest.get(stem, {})
            geom = extract_geom_json_from_dxf(dxf_path)

            raw_texts: List[str] = []
            try:
                doc = ezdxf.readfile(str(dxf_path))
                raw_texts = _extract_raw_text(doc)
            except Exception:
                raw_texts = []

            normalized_texts: List[str] = []
            try:
                if "entities" in geom:
                    normalized_texts = parse_text_content(geom["entities"])  # type: ignore[arg-type]
                else:
                    normalized_texts = list(geom.get("text_content", []))
            except Exception:
                normalized_texts = list(geom.get("text_content", []))
            if not normalized_texts and raw_texts:
                normalized_texts = _normalize_texts(raw_texts)

            raw_joined = "\n".join(raw_texts)
            title_block = parse_title_block(raw_joined) if raw_joined else {}

            preview_path = ""
            if preview_dir and dxf_path in preview_paths:
                preview_file = preview_dir / f"{stem}.png"
                try:
                    render_dxf_to_png(dxf_path, preview_file, config=render_cfg)
                    preview_path = str(preview_file)
                except Exception:
                    preview_path = ""

            writer.writerow(
                {
                    "file_name": dxf_path.name,
                    "source_path": str(dxf_path),
                    "relative_path": meta.get("relative_path", ""),
                    "source_dir": meta.get("source_dir", ""),
                    "suggested_label_cn": meta.get("label_cn", ""),
                    "suggested_label_en": meta.get("label_en", ""),
                    "text_sample": _truncate(raw_texts),
                    "normalized_text_sample": _truncate(normalized_texts),
                    "title_block_drawing_number": title_block.get("drawing_number", ""),
                    "title_block_part_name": title_block.get("part_name", ""),
                    "title_block_material": title_block.get("material", ""),
                    "title_block_scale": title_block.get("scale", ""),
                    "title_block_sheet": title_block.get("sheet", ""),
                    "title_block_date": title_block.get("date", ""),
                    "title_block_weight": title_block.get("weight", ""),
                    "title_block_company": title_block.get("company", ""),
                    "title_block_projection": title_block.get("projection", ""),
                    "title_block_revision": title_block.get("revision", ""),
                    "review_drawing_number": "",
                    "review_part_name": "",
                    "review_material": "",
                    "review_scale": "",
                    "review_sheet": "",
                    "review_date": "",
                    "review_weight": "",
                    "review_company": "",
                    "review_projection": "",
                    "review_revision": "",
                    "dimensions_count": len(geom.get("dimensions", [])),
                    "dimensions_sample": _sample_dimensions(geom.get("dimensions", [])),
                    "layers_count": len(geom.get("layers", [])),
                    "hatches_count": len(geom.get("hatches", [])),
                    "reviewer_label_cn": "",
                    "reviewer_label_en": "",
                    "review_status": "pending",
                    "review_notes": "",
                    "preview_path": preview_path,
                }
            )

    print(f"Wrote {out_path} ({len(sample_paths)} samples)")
    if preview_dir:
        print(f"Rendered {len(preview_paths)} previews to {preview_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
