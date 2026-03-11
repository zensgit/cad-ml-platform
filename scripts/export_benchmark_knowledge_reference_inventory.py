#!/usr/bin/env python3
"""Export benchmark knowledge reference inventory signals."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.benchmark import (  # noqa: E402
    build_knowledge_reference_inventory_status,
    knowledge_reference_inventory_recommendations,
    render_knowledge_reference_inventory_markdown,
)


def _load_json(path_text: str) -> Dict[str, Any]:
    path = Path(path_text).expanduser()
    if not path.exists():
        raise SystemExit(f"JSON input not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected object JSON in {path}")
    return payload


def _write_output(path_text: str, content: str) -> None:
    path = Path(path_text).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_summary(
    *,
    title: str,
    snapshot: Dict[str, Any] | None,
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    component = build_knowledge_reference_inventory_status(snapshot)
    return {
        "title": title,
        "generated_at": int(time.time()),
        "knowledge_reference_inventory": component,
        "recommendations": knowledge_reference_inventory_recommendations(component),
        "artifacts": artifact_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export benchmark standards/tolerance/GD&T reference inventory."
    )
    parser.add_argument(
        "--title",
        default="Benchmark Knowledge Reference Inventory",
    )
    parser.add_argument("--snapshot-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    snapshot = _load_json(args.snapshot_json) if str(args.snapshot_json).strip() else None
    payload = build_summary(
        title=args.title,
        snapshot=snapshot,
        artifact_paths={"snapshot_json": args.snapshot_json},
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    _write_output(args.output_json, rendered + "\n")
    if args.output_md:
        _write_output(
            args.output_md,
            render_knowledge_reference_inventory_markdown(payload, args.title),
        )
    print(rendered)


if __name__ == "__main__":
    main()
