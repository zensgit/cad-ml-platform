#!/usr/bin/env python3
"""Build an HTML review pack for priority DXF samples."""

from __future__ import annotations

import argparse
import csv
import html
import shutil
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.dedupcad_precision.cad_pipeline import DxfRenderConfig, render_dxf_to_png


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _preview_name(index: int, file_name: str) -> str:
    stem = Path(file_name).stem or f"sample_{index:03d}"
    return f"{index + 1:03d}_{stem}.png"


def _render_or_copy(
    source_path: Path, preview_path: Path, output_path: Path, config: DxfRenderConfig
) -> None:
    if preview_path.exists() and preview_path.is_file():
        shutil.copy2(preview_path, output_path)
        return
    render_dxf_to_png(source_path, output_path, config=config)


def _write_html(rows: List[Dict[str, str]], html_path: Path) -> None:
    lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "  <meta charset=\"utf-8\">",
        "  <title>DXF Priority Review Pack</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 20px; }",
        "    table { border-collapse: collapse; width: 100%; }",
        "    th, td { border: 1px solid #ddd; padding: 8px; }",
        "    th { background: #f5f5f5; text-align: left; }",
        "    img { max-width: 360px; height: auto; display: block; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>DXF Priority Review Pack</h1>",
        "  <table>",
        "    <tr>",
        "      <th>#</th>",
        "      <th>File</th>",
        "      <th>Suggested Label</th>",
        "      <th>Graph2D Label</th>",
        "      <th>Confidence</th>",
        "      <th>Preview</th>",
        "    </tr>",
    ]
    for idx, row in enumerate(rows, start=1):
        file_name = html.escape(row.get("file_name", ""))
        suggested = html.escape(row.get("suggested_label_cn", ""))
        graph_label = html.escape(row.get("graph2d_label", ""))
        confidence = html.escape(row.get("graph2d_confidence", ""))
        preview_rel = html.escape(row.get("preview_pack_path", ""))
        if preview_rel:
            preview_cell = f"<img src=\"{preview_rel}\" alt=\"{file_name}\">"
        else:
            preview_cell = "missing"
        lines.extend(
            [
                "    <tr>",
                f"      <td>{idx}</td>",
                f"      <td>{file_name}</td>",
                f"      <td>{suggested}</td>",
                f"      <td>{graph_label}</td>",
                f"      <td>{confidence}</td>",
                f"      <td>{preview_cell}</td>",
                "    </tr>",
            ]
        )
    lines.extend(["  </table>", "</body>", "</html>"])
    html_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a priority review pack.")
    parser.add_argument("--input", required=True, help="Priority CSV input")
    parser.add_argument("--output-dir", required=True, help="Output pack directory")
    parser.add_argument("--render-dpi", type=int, default=200)
    parser.add_argument("--render-size", type=int, default=1024)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    previews_dir = output_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(input_path)
    if not rows:
        print("No rows found; nothing to build.")
        return 1

    config = DxfRenderConfig(size_px=args.render_size, dpi=args.render_dpi)

    for idx, row in enumerate(rows):
        source_path = Path(row.get("source_path", ""))
        preview_raw = (row.get("preview_path") or "").strip()
        preview_path = Path(preview_raw) if preview_raw else Path()
        preview_name = _preview_name(idx, row.get("file_name", ""))
        output_preview = previews_dir / preview_name
        if source_path.exists():
            _render_or_copy(source_path, preview_path, output_preview, config)
            row["preview_pack_path"] = f"previews/{preview_name}"
        else:
            row["preview_pack_path"] = ""

    csv_path = output_dir / "review_priority_pack.csv"
    fieldnames = list(rows[0].keys())
    if "preview_pack_path" not in fieldnames:
        fieldnames.append("preview_pack_path")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _write_html(rows, output_dir / "index.html")
    print(f"Wrote pack to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
