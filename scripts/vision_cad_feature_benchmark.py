from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.vision_analyzer import VisionAnalyzer


def _parse_thresholds(raw_items: List[str]) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid threshold override: {item}")
        key, value = item.split("=", 1)
        thresholds[key.strip()] = float(value)
    return thresholds


def _build_synthetic_samples() -> List[Tuple[str, Image.Image]]:
    samples: List[Tuple[str, Image.Image]] = []

    horizontal = Image.new("L", (120, 80), color=255)
    draw = ImageDraw.Draw(horizontal)
    draw.line((10, 10, 110, 10), fill=0, width=3)
    samples.append(("horizontal_line", horizontal))

    diagonal = Image.new("L", (100, 100), color=255)
    draw = ImageDraw.Draw(diagonal)
    draw.line((10, 90, 90, 10), fill=0, width=3)
    samples.append(("diagonal_line", diagonal))

    circle = Image.new("L", (100, 100), color=255)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((30, 30, 70, 70), outline=0, width=3)
    samples.append(("circle", circle))

    arc = Image.new("L", (100, 100), color=255)
    draw = ImageDraw.Draw(arc)
    draw.arc((20, 20, 80, 80), start=0, end=180, fill=0, width=4)
    samples.append(("arc", arc))

    return samples


def _load_images(input_dir: Path, max_samples: int | None) -> List[Tuple[str, Image.Image]]:
    patterns = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    samples: List[Tuple[str, Image.Image]] = []
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() not in patterns:
            continue
        image = Image.open(path).convert("L")
        samples.append((path.name, image))
        if max_samples and len(samples) >= max_samples:
            break
    return samples


async def _run(samples: List[Tuple[str, Image.Image]], thresholds: Dict[str, float]) -> List[Dict]:
    analyzer = VisionAnalyzer()
    results = []
    for name, image in samples:
        features = await analyzer._extract_cad_features(image, thresholds)
        drawings = features.get("drawings", {})
        stats = features.get("stats", {})
        results.append(
            {
                "name": name,
                "lines": len(drawings.get("lines", [])),
                "circles": len(drawings.get("circles", [])),
                "arcs": len(drawings.get("arcs", [])),
                "ink_ratio": stats.get("ink_ratio"),
                "components": stats.get("components"),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CAD feature heuristics.")
    parser.add_argument("--input-dir", type=Path, help="Directory with raster CAD images")
    parser.add_argument("--output-json", type=Path, help="Optional JSON output file")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples loaded")
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="Override thresholds, e.g. line_aspect=5.0 (repeatable)",
    )
    args = parser.parse_args()

    thresholds = _parse_thresholds(args.threshold)
    if args.input_dir:
        samples = _load_images(args.input_dir, args.max_samples)
    else:
        samples = _build_synthetic_samples()
        if args.max_samples:
            samples = samples[: args.max_samples]

    results = asyncio.run(_run(samples, thresholds))
    for result in results:
        print(
            f"{result['name']}: lines={result['lines']} circles={result['circles']} "
            f"arcs={result['arcs']} ink_ratio={result['ink_ratio']}"
        )
    print(f"total_samples={len(results)}")

    if args.output_json:
        payload = {"thresholds": thresholds, "results": results}
        args.output_json.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
