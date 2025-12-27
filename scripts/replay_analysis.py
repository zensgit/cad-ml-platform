#!/usr/bin/env python3
"""
Replay analysis on a list of CAD files and store JSONL results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import httpx


def _load_inputs(path: Path) -> List[Path]:
    items: List[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(Path(line))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay CAD analysis on a file list")
    parser.add_argument("--input-list", required=True, help="Text file with CAD paths")
    parser.add_argument("--api-url", default="http://localhost:8000/api/v1", help="Base API URL")
    parser.add_argument("--output-dir", default="reports/replay", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without API calls")
    args = parser.parse_args()

    input_list = Path(args.input_list)
    if not input_list.exists():
        raise SystemExit(f"Input list not found: {input_list}")

    inputs = _load_inputs(input_list)
    if args.dry_run:
        print(f"dry_run=true inputs={len(inputs)}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "replay_results.jsonl"
    summary_path = output_dir / "summary.json"

    options = {
        "extract_features": True,
        "classify_parts": True,
        "quality_check": True,
        "process_recommendation": True,
        "estimate_cost": True,
    }

    stats = {"total": 0, "success": 0, "error": 0}
    with results_path.open("w", encoding="utf-8") as out:
        for item in inputs:
            stats["total"] += 1
            if not item.exists():
                out.write(json.dumps({"path": str(item), "status": "missing"}) + "\n")
                stats["error"] += 1
                continue

            with item.open("rb") as f:
                files = {"file": (item.name, f.read(), "application/octet-stream")}
            data = {"options": json.dumps(options)}

            try:
                res = httpx.post(f"{args.api_url}/analyze/", files=files, data=data, timeout=60.0)
                if res.status_code == 200:
                    out.write(json.dumps({"path": str(item), "status": "ok", "result": res.json()}) + "\n")
                    stats["success"] += 1
                else:
                    out.write(json.dumps({"path": str(item), "status": "error", "error": res.text}) + "\n")
                    stats["error"] += 1
            except Exception as exc:
                out.write(json.dumps({"path": str(item), "status": "error", "error": str(exc)}) + "\n")
                stats["error"] += 1

    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote: {results_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
