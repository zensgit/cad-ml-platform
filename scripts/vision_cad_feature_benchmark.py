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


def _coerce_threshold_mapping(raw: object, context: str) -> Dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a mapping of numeric values")
    thresholds: Dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"{context}[{key}] must be numeric")
        thresholds[str(key)] = float(value)
    return thresholds


def _coerce_grid_mapping(raw: object) -> Dict[str, List[float]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("grid must be a mapping of numeric lists")
    grid: Dict[str, List[float]] = {}
    for key, value in raw.items():
        values: List[float]
        if isinstance(value, (int, float)):
            values = [float(value)]
        elif isinstance(value, list):
            if not value:
                raise ValueError(f"grid[{key}] must not be empty")
            values = []
            for item in value:
                if not isinstance(item, (int, float)):
                    raise ValueError(f"grid[{key}] values must be numeric")
                values.append(float(item))
        else:
            raise ValueError(f"grid[{key}] must be a list of numeric values")
        grid[str(key)] = values
    return grid


def _load_threshold_file(
    path: Path,
) -> tuple[Dict[str, float], Dict[str, List[float]], List[Dict[str, float]]]:
    if not path.exists():
        raise FileNotFoundError(f"threshold file not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError("PyYAML is required to load YAML threshold files") from exc
        payload = yaml.safe_load(path.read_text())
    else:
        payload = json.loads(path.read_text())

    if payload is None:
        return {}, {}, []
    if isinstance(payload, list):
        variants = [
            _coerce_threshold_mapping(item, f"variants[{idx}]")
            for idx, item in enumerate(payload)
        ]
        return {}, {}, variants
    if not isinstance(payload, dict):
        raise ValueError("threshold file must be a mapping or list of mappings")

    variants_raw = payload.get("variants")
    grid_raw = payload.get("grid")
    thresholds_raw = payload.get("thresholds")
    if thresholds_raw is None:
        thresholds_raw = {
            key: value
            for key, value in payload.items()
            if key not in {"grid", "variants"}
        }

    thresholds = _coerce_threshold_mapping(thresholds_raw, "thresholds")
    grid = _coerce_grid_mapping(grid_raw)
    variants: List[Dict[str, float]] = []
    if variants_raw is not None:
        if not isinstance(variants_raw, list):
            raise ValueError("variants must be a list of mappings")
        variants = [
            _coerce_threshold_mapping(item, f"variants[{idx}]")
            for idx, item in enumerate(variants_raw)
        ]
    return thresholds, grid, variants


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


async def _run(
    samples: List[Tuple[str, Image.Image]],
    thresholds: Dict[str, float],
    initialize_clients: bool = True,
) -> List[Dict]:
    analyzer = VisionAnalyzer(initialize_clients=initialize_clients)
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


def _write_compare_csv(output_path: Path, comparison: Dict) -> None:
    fieldnames = [
        "combo_index",
        "status",
        "total_lines_delta",
        "total_circles_delta",
        "total_arcs_delta",
        "avg_ink_ratio_delta",
        "avg_components_delta",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for combo in comparison.get("combo_deltas", []):
            status = "missing_baseline" if combo.get("missing_baseline") else "ok"
            summary_delta = combo.get("summary_delta", {}) if status == "ok" else {}
            writer.writerow(
                {
                    "combo_index": combo.get("combo_index"),
                    "status": status,
                    "total_lines_delta": summary_delta.get("total_lines"),
                    "total_circles_delta": summary_delta.get("total_circles"),
                    "total_arcs_delta": summary_delta.get("total_arcs"),
                    "avg_ink_ratio_delta": summary_delta.get("avg_ink_ratio"),
                    "avg_components_delta": summary_delta.get("avg_components"),
                }
            )


def _compare_results(current: List[Dict], baseline: List[Dict]) -> Dict:
    comparison: Dict[str, List[Dict]] = {"combo_deltas": []}
    for idx, entry in enumerate(current, start=1):
        baseline_entry = baseline[idx - 1] if idx - 1 < len(baseline) else None
        if baseline_entry is None:
            comparison["combo_deltas"].append(
                {"combo_index": idx, "missing_baseline": True}
            )
            continue

        summary_delta: Dict[str, float] = {}
        for key, value in entry.get("summary", {}).items():
            base_value = baseline_entry.get("summary", {}).get(key)
            if isinstance(value, (int, float)) and isinstance(base_value, (int, float)):
                delta = value - base_value
                summary_delta[key] = round(delta, 4)

        base_samples = {
            sample.get("name"): sample for sample in baseline_entry.get("samples", [])
        }
        sample_deltas: List[Dict[str, float]] = []
        missing_samples: List[str] = []
        for sample in entry.get("samples", []):
            name = sample.get("name")
            base = base_samples.get(name)
            if base is None:
                if name:
                    missing_samples.append(name)
                continue
            sample_deltas.append(
                {
                    "name": name,
                    "lines_delta": sample.get("lines", 0) - base.get("lines", 0),
                    "circles_delta": sample.get("circles", 0) - base.get("circles", 0),
                    "arcs_delta": sample.get("arcs", 0) - base.get("arcs", 0),
                    "ink_ratio_delta": round(
                        sample.get("ink_ratio", 0.0) - base.get("ink_ratio", 0.0), 4
                    ),
                    "components_delta": sample.get("components", 0)
                    - base.get("components", 0),
                }
            )

        comparison["combo_deltas"].append(
            {
                "combo_index": idx,
                "summary_delta": summary_delta,
                "sample_deltas": sample_deltas,
                "missing_samples": missing_samples,
            }
        )

    return comparison


def _print_comparison(comparison: Dict) -> None:
    for combo in comparison.get("combo_deltas", []):
        index = combo.get("combo_index")
        if combo.get("missing_baseline"):
            print(f"compare combo={index}: missing baseline")
            continue
        print(f"compare combo={index} summary_delta={combo.get('summary_delta')}")
        for sample in combo.get("sample_deltas", []):
            print(
                "  compare sample={name} lines={lines_delta} circles={circles_delta} "
                "arcs={arcs_delta} ink_ratio={ink_ratio_delta} components={components_delta}".format(
                    **sample
                )
            )
        missing = combo.get("missing_samples") or []
        if missing:
            print(f"  compare missing_samples={missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CAD feature heuristics.")
    parser.add_argument("--input-dir", type=Path, help="Directory with raster CAD images")
    parser.add_argument("--output-json", type=Path, help="Optional JSON output file")
    parser.add_argument("--output-csv", type=Path, help="Optional CSV output file")
    parser.add_argument(
        "--output-compare-csv",
        type=Path,
        help="Optional CSV output for compare summary deltas (requires --compare-json)",
    )
    parser.add_argument("--compare-json", type=Path, help="Baseline JSON to compare against")
    parser.add_argument(
        "--threshold-file",
        type=Path,
        help="JSON/YAML file with thresholds, grid, or variants",
    )
    parser.add_argument("--max-samples", type=int, help="Limit number of samples loaded")
    parser.add_argument(
        "--no-clients",
        action="store_true",
        help="Skip initializing external vision clients during benchmarking",
    )
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

    if args.output_compare_csv and not args.compare_json:
        print("error: --output-compare-csv requires --compare-json", file=sys.stderr)
        sys.exit(1)

    file_thresholds: Dict[str, float] = {}
    file_grid: Dict[str, List[float]] = {}
    file_variants: List[Dict[str, float]] = []
    if args.threshold_file:
        file_thresholds, file_grid, file_variants = _load_threshold_file(
            args.threshold_file
        )

    cli_thresholds = _parse_thresholds(args.threshold)
    cli_grid = _parse_grid(args.grid)
    thresholds = {**file_thresholds, **cli_thresholds}
    grid = {**file_grid, **cli_grid}
    if args.input_dir:
        samples = _load_images(args.input_dir, args.max_samples)
    else:
        samples = _build_synthetic_samples()
        if args.max_samples:
            samples = samples[: args.max_samples]

    if file_variants:
        if grid:
            print(
                "note: ignoring grid overrides because threshold-file provides variants",
                file=sys.stderr,
            )
        variants = []
        for variant in file_variants:
            merged = dict(file_thresholds)
            merged.update(variant)
            merged.update(cli_thresholds)
            variants.append(merged)
    else:
        variants = _build_grid_variants(thresholds, grid)
    grid_results = []
    for idx, variant in enumerate(variants, start=1):
        results = asyncio.run(_run(samples, variant, initialize_clients=not args.no_clients))
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

    comparison = None
    if args.compare_json:
        compare_payload = json.loads(args.compare_json.read_text())
        comparison = _compare_results(grid_results, compare_payload.get("results", []))
        _print_comparison(comparison)

    if args.output_json:
        payload = {
            "base_thresholds": thresholds,
            "grid": grid,
            "results": grid_results,
        }
        if comparison is not None:
            payload["comparison"] = comparison
        args.output_json.write_text(json.dumps(payload, indent=2))
    if args.output_csv:
        grid_keys = sorted(grid.keys())
        _write_csv(args.output_csv, grid_results, grid_keys)
    if args.output_compare_csv and comparison is not None:
        _write_compare_csv(args.output_compare_csv, comparison)


if __name__ == "__main__":
    main()
