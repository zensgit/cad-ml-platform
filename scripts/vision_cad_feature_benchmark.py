from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


def _parse_grid(raw_items: List[str]) -> Dict[str, List[float]]:
    grid: Dict[str, List[float]] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid grid override: {item}")
        key, value = item.split("=", 1)
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
        if not values:
            raise ValueError(f"Grid override has no values: {item}")
        grid[key.strip()] = values
    return grid


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


def _build_grid_variants(
    base_thresholds: Dict[str, float], grid: Dict[str, List[float]]
) -> List[Dict[str, float]]:
    if not grid:
        return [dict(base_thresholds)]

    keys = sorted(grid.keys())
    variants: List[Dict[str, float]] = []
    import itertools

    for values in itertools.product(*(grid[key] for key in keys)):
        thresholds = dict(base_thresholds)
        for key, value in zip(keys, values):
            thresholds[key] = value
        variants.append(thresholds)
    return variants


def _summarize(results: List[Dict]) -> Dict[str, float | int | None]:
    if not results:
        return {
            "total_lines": 0,
            "total_circles": 0,
            "total_arcs": 0,
            "avg_ink_ratio": None,
            "avg_components": None,
        }
    total_lines = sum(item["lines"] for item in results)
    total_circles = sum(item["circles"] for item in results)
    total_arcs = sum(item["arcs"] for item in results)
    avg_ink_ratio = sum(item["ink_ratio"] for item in results) / len(results)
    avg_components = sum(item["components"] for item in results) / len(results)
    return {
        "total_lines": int(total_lines),
        "total_circles": int(total_circles),
        "total_arcs": int(total_arcs),
        "avg_ink_ratio": round(avg_ink_ratio, 4),
        "avg_components": round(avg_components, 2),
    }


def _write_csv(
    output_path: Path,
    grid_results: List[Dict],
    grid_keys: List[str],
) -> None:
    fieldnames = [
        "combo_index",
        "sample",
        "lines",
        "circles",
        "arcs",
        "ink_ratio",
        "components",
    ] + grid_keys
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, entry in enumerate(grid_results, start=1):
            thresholds = entry["thresholds"]
            for sample in entry["samples"]:
                row = {
                    "combo_index": idx,
                    "sample": sample["name"],
                    "lines": sample["lines"],
                    "circles": sample["circles"],
                    "arcs": sample["arcs"],
                    "ink_ratio": sample["ink_ratio"],
                    "components": sample["components"],
                }
                for key in grid_keys:
                    row[key] = thresholds.get(key)
                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CAD feature heuristics.")
    parser.add_argument("--input-dir", type=Path, help="Directory with raster CAD images")
    parser.add_argument("--output-json", type=Path, help="Optional JSON output file")
    parser.add_argument("--output-csv", type=Path, help="Optional CSV output file")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples loaded")
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="Override thresholds, e.g. line_aspect=5.0 (repeatable)",
    )
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        help="Grid overrides, e.g. line_aspect=4,5,6 (repeatable)",
    )
    args = parser.parse_args()

    thresholds = _parse_thresholds(args.threshold)
    grid = _parse_grid(args.grid)
    if args.input_dir:
        samples = _load_images(args.input_dir, args.max_samples)
    else:
        samples = _build_synthetic_samples()
        if args.max_samples:
            samples = samples[: args.max_samples]

    variants = _build_grid_variants(thresholds, grid)
    grid_results = []
    for idx, variant in enumerate(variants, start=1):
        results = asyncio.run(_run(samples, variant))
        grid_results.append(
            {"thresholds": variant, "samples": results, "summary": _summarize(results)}
        )
        label = f"combo={idx}/{len(variants)} thresholds={variant}"
        print(label)
        for result in results:
            print(
                f"  {result['name']}: lines={result['lines']} circles={result['circles']} "
                f"arcs={result['arcs']} ink_ratio={result['ink_ratio']}"
            )
        print(f"  summary={grid_results[-1]['summary']}")
    print(f"total_samples={len(samples)} total_combos={len(variants)}")

    if args.output_json:
        payload = {
            "base_thresholds": thresholds,
            "grid": grid,
            "results": grid_results,
        }
        args.output_json.write_text(json.dumps(payload, indent=2))
    if args.output_csv:
        grid_keys = sorted(grid.keys())
        _write_csv(args.output_csv, grid_results, grid_keys)


if __name__ == "__main__":
    main()
