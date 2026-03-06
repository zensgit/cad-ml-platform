#!/usr/bin/env python3
"""Validate online example H5/STEP inputs against local AI entrypoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def inspect_h5_input(path: Path) -> Dict[str, Any]:
    from src.ml.history_sequence_classifier import HistorySequenceClassifier
    from src.ml.history_sequence_tools import load_command_tokens_from_h5

    payload: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        payload["status"] = "missing"
        return payload

    with h5py.File(path, "r") as handle:
        payload["keys"] = list(handle.keys())
        if "vec" in handle:
            payload["vec_shape"] = list(handle["vec"].shape)
            payload["vec_dtype"] = str(handle["vec"].dtype)

    tokens = load_command_tokens_from_h5(str(path))
    payload["tokens_length"] = len(tokens)
    payload["tokens_head"] = [int(x) for x in tokens[:10]]

    classifier = HistorySequenceClassifier()
    payload["prediction"] = classifier.predict_from_h5_file(str(path))
    payload["status"] = "ok"
    return payload


def inspect_step_input(path: Path) -> Dict[str, Any]:
    from src.core.geometry.engine import HAS_OCC, get_geometry_engine

    payload: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "has_occ": bool(HAS_OCC),
    }
    if not path.exists():
        payload["status"] = "missing"
        return payload
    if not HAS_OCC:
        payload["status"] = "skipped_no_occ"
        return payload

    engine = get_geometry_engine()
    shape = engine.load_step(path.read_bytes(), path.name)
    payload["shape_loaded"] = shape is not None
    if shape is None:
        payload["status"] = "load_failed"
        return payload

    brep_features = engine.extract_brep_features(shape)
    brep_graph = engine.extract_brep_graph(shape)
    payload["brep_features"] = brep_features
    payload["brep_graph"] = {
        "valid_3d": brep_graph.get("valid_3d"),
        "graph_schema_version": brep_graph.get("graph_schema_version"),
        "node_count": brep_graph.get("node_count"),
        "edge_count": brep_graph.get("edge_count"),
        "graph_metadata": brep_graph.get("graph_metadata"),
    }
    payload["status"] = "ok"
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate online example H5/STEP inputs against local AI entrypoints."
    )
    parser.add_argument(
        "--h5-file",
        default=(
            "/private/tmp/cad-ai-example-data-20260307/"
            "HPSketch/data/0000/00000007_1.h5"
        ),
        help="Path to a sample HPSketch .h5 file.",
    )
    parser.add_argument(
        "--step-file",
        default="/private/tmp/cad-ai-example-data-20260307/foxtrot/examples/cube_hole.step",
        help="Path to a sample STEP file.",
    )
    parser.add_argument(
        "--output",
        default=(
            f"reports/experiments/{time.strftime('%Y%m%d')}/"
            "online_example_ai_inputs_validation.json"
        ),
        help="JSON report output path.",
    )
    args = parser.parse_args(argv)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cwd": str(REPO_ROOT),
        "h5_validation": inspect_h5_input(Path(args.h5_file)),
        "step_validation": inspect_step_input(Path(args.step_file)),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
