#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("manifest must be a JSON list")
    return [item for item in payload if isinstance(item, dict)]


def _safe_filename(value: str, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    text = text.replace("/", "_").replace("\\", "_")
    return text if text.lower().endswith(".dxf") else f"{text}.dxf"


def _build_single_file(path: Path, idx: int) -> None:
    import ezdxf  # noqa: WPS433

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    size = 100 + (idx % 7) * 15
    msp.add_line((0, 0), (size, 0))
    msp.add_line((size, 0), (size, size))
    msp.add_line((size, size), (0, size))
    msp.add_line((0, size), (0, 0))
    if idx % 2 == 0:
        msp.add_circle((size / 2, size / 2), max(5, size / 4))
    else:
        msp.add_arc((size / 2, size / 2), max(5, size / 3), start_angle=0, end_angle=180)

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(path)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build synthetic DXF dataset for hybrid blind benchmarking.")
    parser.add_argument(
        "--manifest",
        default="tests/golden/golden_dxf_hybrid_cases.json",
        help="Golden manifest JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/hybrid_blind_synth",
        help="Output directory for generated DXF files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional max files limit (0 means all).",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_manifest(manifest_path)
    if args.max_files and args.max_files > 0:
        cases = cases[: int(args.max_files)]

    generated = 0
    for idx, case in enumerate(cases):
        file_name = _safe_filename(str(case.get("filename") or ""), f"case_{idx:04d}.dxf")
        target = output_dir / file_name
        _build_single_file(target, idx)
        generated += 1

    print(f"output_dir={output_dir}")
    print(f"generated={generated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
