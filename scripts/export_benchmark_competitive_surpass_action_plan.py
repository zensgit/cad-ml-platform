#!/usr/bin/env python3
"""Export benchmark competitive-surpass action plan signals."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.benchmark import (  # noqa: E402
    build_competitive_surpass_action_plan,
    competitive_surpass_action_plan_recommendations,
    render_competitive_surpass_action_plan_markdown,
)


def _load_json(path_text: str) -> Dict[str, Any]:
    path = Path(path_text).expanduser()
    if not path.exists():
        raise SystemExit(f"JSON input not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
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


def build_summary(
    *,
    title: str,
    benchmark_competitive_surpass_index: Dict[str, Any],
    benchmark_competitive_surpass_trend: Dict[str, Any],
    benchmark_engineering_signals: Dict[str, Any],
    benchmark_knowledge_domain_action_plan: Dict[str, Any],
    benchmark_realdata_signals: Dict[str, Any],
    benchmark_realdata_scorecard: Dict[str, Any],
    benchmark_operator_adoption: Dict[str, Any],
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    component = build_competitive_surpass_action_plan(
        benchmark_competitive_surpass_index=benchmark_competitive_surpass_index,
        benchmark_competitive_surpass_trend=benchmark_competitive_surpass_trend,
        benchmark_engineering_signals=benchmark_engineering_signals,
        benchmark_knowledge_domain_action_plan=benchmark_knowledge_domain_action_plan,
        benchmark_realdata_signals=benchmark_realdata_signals,
        benchmark_realdata_scorecard=benchmark_realdata_scorecard,
        benchmark_operator_adoption=benchmark_operator_adoption,
    )
    return {
        "title": title,
        "generated_at": int(time.time()),
        "competitive_surpass_action_plan": component,
        "recommendations": competitive_surpass_action_plan_recommendations(component),
        "artifacts": artifact_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export benchmark competitive-surpass action plan signals."
    )
    parser.add_argument(
        "--title",
        default="Benchmark Competitive Surpass Action Plan",
    )
    parser.add_argument("--benchmark-competitive-surpass-index", default="")
    parser.add_argument("--benchmark-competitive-surpass-trend", default="")
    parser.add_argument("--benchmark-engineering-signals", default="")
    parser.add_argument("--benchmark-knowledge-domain-action-plan", default="")
    parser.add_argument("--benchmark-realdata-signals", default="")
    parser.add_argument("--benchmark-realdata-scorecard", default="")
    parser.add_argument("--benchmark-operator-adoption", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    payload = build_summary(
        title=args.title,
        benchmark_competitive_surpass_index=_maybe_load_json(
            args.benchmark_competitive_surpass_index
        ),
        benchmark_competitive_surpass_trend=_maybe_load_json(
            args.benchmark_competitive_surpass_trend
        ),
        benchmark_engineering_signals=_maybe_load_json(
            args.benchmark_engineering_signals
        ),
        benchmark_knowledge_domain_action_plan=_maybe_load_json(
            args.benchmark_knowledge_domain_action_plan
        ),
        benchmark_realdata_signals=_maybe_load_json(args.benchmark_realdata_signals),
        benchmark_realdata_scorecard=_maybe_load_json(
            args.benchmark_realdata_scorecard
        ),
        benchmark_operator_adoption=_maybe_load_json(args.benchmark_operator_adoption),
        artifact_paths={
            "benchmark_competitive_surpass_index": (
                args.benchmark_competitive_surpass_index
            ),
            "benchmark_competitive_surpass_trend": (
                args.benchmark_competitive_surpass_trend
            ),
            "benchmark_engineering_signals": args.benchmark_engineering_signals,
            "benchmark_knowledge_domain_action_plan": (
                args.benchmark_knowledge_domain_action_plan
            ),
            "benchmark_realdata_signals": args.benchmark_realdata_signals,
            "benchmark_realdata_scorecard": args.benchmark_realdata_scorecard,
            "benchmark_operator_adoption": args.benchmark_operator_adoption,
        },
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        _write_output(args.output_json, rendered + "\n")
    if args.output_md:
        _write_output(
            args.output_md,
            render_competitive_surpass_action_plan_markdown(payload, args.title),
        )
    print(rendered)


if __name__ == "__main__":
    main()
