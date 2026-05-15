#!/usr/bin/env python3
"""Export the CAD ML forward scorecard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.benchmark.forward_scorecard import (  # noqa: E402
    build_forward_scorecard,
    render_forward_scorecard_markdown,
)
from src.models.readiness_registry import build_model_readiness_snapshot  # noqa: E402


def _load_json(path_text: str) -> Dict[str, Any]:
    path = Path(path_text).expanduser()
    if not path.exists():
        raise SystemExit(f"JSON input not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected object JSON in {path}")
    return payload


def _maybe_load_json(path_text: str) -> Dict[str, Any]:
    if not str(path_text or "").strip():
        return {}
    return _load_json(path_text)


def _write_output(path_text: str, content: str) -> None:
    output_path = Path(path_text).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the CAD ML forward scorecard.")
    parser.add_argument("--title", default="CAD ML Forward Scorecard")
    parser.add_argument("--model-readiness-summary", default="")
    parser.add_argument("--hybrid-summary", default="")
    parser.add_argument("--graph2d-summary", default="")
    parser.add_argument("--history-summary", default="")
    parser.add_argument("--brep-summary", default="")
    parser.add_argument("--qdrant-summary", default="")
    parser.add_argument("--review-queue-summary", default="")
    parser.add_argument("--knowledge-summary", default="")
    parser.add_argument("--manufacturing-evidence-summary", default="")
    parser.add_argument("--manufacturing-review-manifest-validation-summary", default="")
    parser.add_argument(
        "--output-json",
        default="reports/benchmark/forward_scorecard/latest.json",
    )
    parser.add_argument(
        "--output-md",
        default="reports/benchmark/forward_scorecard/latest.md",
    )
    args = parser.parse_args()

    if args.model_readiness_summary:
        model_readiness = _load_json(args.model_readiness_summary)
    else:
        model_readiness = build_model_readiness_snapshot().to_dict()

    artifact_paths = {
        "model_readiness_summary": args.model_readiness_summary,
        "hybrid_summary": args.hybrid_summary,
        "graph2d_summary": args.graph2d_summary,
        "history_summary": args.history_summary,
        "brep_summary": args.brep_summary,
        "qdrant_summary": args.qdrant_summary,
        "review_queue_summary": args.review_queue_summary,
        "knowledge_summary": args.knowledge_summary,
        "manufacturing_evidence_summary": args.manufacturing_evidence_summary,
        "manufacturing_review_manifest_validation_summary": (
            args.manufacturing_review_manifest_validation_summary
        ),
    }
    payload = build_forward_scorecard(
        title=args.title,
        model_readiness=model_readiness,
        hybrid_summary=_maybe_load_json(args.hybrid_summary),
        graph2d_summary=_maybe_load_json(args.graph2d_summary),
        history_summary=_maybe_load_json(args.history_summary),
        brep_summary=_maybe_load_json(args.brep_summary),
        qdrant_summary=_maybe_load_json(args.qdrant_summary),
        review_queue_summary=_maybe_load_json(args.review_queue_summary),
        knowledge_summary=_maybe_load_json(args.knowledge_summary),
        manufacturing_summary=_maybe_load_json(args.manufacturing_evidence_summary),
        manufacturing_review_manifest_validation=_maybe_load_json(
            args.manufacturing_review_manifest_validation_summary
        ),
        artifact_paths={k: v for k, v in artifact_paths.items() if v},
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        _write_output(args.output_json, rendered + "\n")
    if args.output_md:
        _write_output(
            args.output_md,
            render_forward_scorecard_markdown(payload, args.title),
        )
    print(rendered)


if __name__ == "__main__":
    main()
