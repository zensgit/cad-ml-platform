#!/usr/bin/env python3
"""Quality gate for Graph2D hybrid review-pack summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "min_total_rows": 10,
    "max_candidate_rate": 0.7,
    "max_hybrid_rejected_rate": 0.6,
    "max_conflict_rate": 0.5,
    "max_low_confidence_rate": 0.7,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload


def _load_yaml_defaults(config_path: str, section: str) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    section_payload = payload.get(section)
    if not isinstance(section_payload, dict):
        return {}
    return section_payload


def _resolve_thresholds(
    config_payload: Dict[str, Any],
    cli_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    out = dict(DEFAULT_THRESHOLDS)
    for key in DEFAULT_THRESHOLDS:
        if key in config_payload:
            out[key] = config_payload.get(key)
    for key, value in cli_overrides.items():
        if value is not None:
            out[key] = value
    return {
        "min_total_rows": _safe_int(
            out.get("min_total_rows"), DEFAULT_THRESHOLDS["min_total_rows"]
        ),
        "max_candidate_rate": _safe_float(
            out.get("max_candidate_rate"), DEFAULT_THRESHOLDS["max_candidate_rate"]
        ),
        "max_hybrid_rejected_rate": _safe_float(
            out.get("max_hybrid_rejected_rate"),
            DEFAULT_THRESHOLDS["max_hybrid_rejected_rate"],
        ),
        "max_conflict_rate": _safe_float(
            out.get("max_conflict_rate"), DEFAULT_THRESHOLDS["max_conflict_rate"]
        ),
        "max_low_confidence_rate": _safe_float(
            out.get("max_low_confidence_rate"),
            DEFAULT_THRESHOLDS["max_low_confidence_rate"],
        ),
    }


def evaluate_review_pack_gate(
    summary: Dict[str, Any], thresholds: Dict[str, Any]
) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []

    total_rows = _safe_int(summary.get("total_rows"), 0)
    candidate_rows = _safe_int(summary.get("candidate_rows"), 0)
    hybrid_rejected_count = _safe_int(summary.get("hybrid_rejected_count"), 0)
    conflict_count = _safe_int(summary.get("conflict_count"), 0)
    low_confidence_count = _safe_int(summary.get("low_confidence_count"), 0)

    if total_rows < int(thresholds["min_total_rows"]):
        warnings.append(
            f"total_rows {total_rows} < min_total_rows {int(thresholds['min_total_rows'])}"
        )

    denominator = total_rows if total_rows > 0 else 1
    candidate_rate = float(candidate_rows) / float(denominator)
    hybrid_rejected_rate = float(hybrid_rejected_count) / float(denominator)
    conflict_rate = float(conflict_count) / float(denominator)
    low_confidence_rate = float(low_confidence_count) / float(denominator)

    if candidate_rate > float(thresholds["max_candidate_rate"]):
        failures.append(
            f"candidate_rate {candidate_rate:.4f} > max_candidate_rate "
            f"{float(thresholds['max_candidate_rate']):.4f}"
        )
    if hybrid_rejected_rate > float(thresholds["max_hybrid_rejected_rate"]):
        failures.append(
            f"hybrid_rejected_rate {hybrid_rejected_rate:.4f} > "
            f"max_hybrid_rejected_rate {float(thresholds['max_hybrid_rejected_rate']):.4f}"
        )
    if conflict_rate > float(thresholds["max_conflict_rate"]):
        failures.append(
            f"conflict_rate {conflict_rate:.4f} > max_conflict_rate "
            f"{float(thresholds['max_conflict_rate']):.4f}"
        )
    if low_confidence_rate > float(thresholds["max_low_confidence_rate"]):
        failures.append(
            f"low_confidence_rate {low_confidence_rate:.4f} > max_low_confidence_rate "
            f"{float(thresholds['max_low_confidence_rate']):.4f}"
        )

    status = "passed" if not failures else "failed"
    return {
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "metrics": {
            "total_rows": total_rows,
            "candidate_rows": candidate_rows,
            "hybrid_rejected_count": hybrid_rejected_count,
            "conflict_count": conflict_count,
            "low_confidence_count": low_confidence_count,
            "candidate_rate": round(candidate_rate, 6),
            "hybrid_rejected_rate": round(hybrid_rejected_rate, 6),
            "conflict_rate": round(conflict_rate, 6),
            "low_confidence_rate": round(low_confidence_rate, 6),
        },
        "thresholds": thresholds,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Graph2D review-pack quality gate.")
    parser.add_argument("--summary-json", required=True, help="Review-pack summary JSON path.")
    parser.add_argument(
        "--config",
        default="config/graph2d_review_pack_gate.yaml",
        help="Optional YAML config path.",
    )
    parser.add_argument("--output", default="", help="Optional report JSON output path.")
    parser.add_argument("--min-total-rows", type=int, default=None)
    parser.add_argument("--max-candidate-rate", type=float, default=None)
    parser.add_argument("--max-hybrid-rejected-rate", type=float, default=None)
    parser.add_argument("--max-conflict-rate", type=float, default=None)
    parser.add_argument("--max-low-confidence-rate", type=float, default=None)
    args = parser.parse_args(argv)

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        raise SystemExit(f"summary not found: {summary_path}")

    config_payload = _load_yaml_defaults(args.config, "graph2d_review_pack_gate")
    thresholds = _resolve_thresholds(
        config_payload=config_payload,
        cli_overrides={
            "min_total_rows": args.min_total_rows,
            "max_candidate_rate": args.max_candidate_rate,
            "max_hybrid_rejected_rate": args.max_hybrid_rejected_rate,
            "max_conflict_rate": args.max_conflict_rate,
            "max_low_confidence_rate": args.max_low_confidence_rate,
        },
    )
    summary = _read_json(summary_path)
    report = evaluate_review_pack_gate(summary=summary, thresholds=thresholds)

    output_path = Path(args.output) if args.output else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
