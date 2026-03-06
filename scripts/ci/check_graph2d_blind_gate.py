#!/usr/bin/env python3
"""Graph2D blind-evaluation quality gate.

This gate consumes the summary produced by `scripts/diagnose_graph2d_on_dxf_dir.py`
with `--strip-text-entities --mask-filename` and enforces minimum quality bars.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "min_accuracy": 0.25,
    "max_top_pred_ratio": 0.65,
    "min_distinct_pred_labels": 5,
    "max_low_conf_rate": 0.8,
    "low_conf_threshold": 0.2,
    "require_strip_text_entities": True,
    "require_mask_filename": True,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


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
    for key in DEFAULT_THRESHOLDS.keys():
        if key in config_payload:
            out[key] = config_payload.get(key)
    for key, value in cli_overrides.items():
        if value is not None:
            out[key] = value
    return {
        "min_accuracy": _safe_float(out.get("min_accuracy"), DEFAULT_THRESHOLDS["min_accuracy"]),
        "max_top_pred_ratio": _safe_float(
            out.get("max_top_pred_ratio"), DEFAULT_THRESHOLDS["max_top_pred_ratio"]
        ),
        "min_distinct_pred_labels": _safe_int(
            out.get("min_distinct_pred_labels"),
            DEFAULT_THRESHOLDS["min_distinct_pred_labels"],
        ),
        "max_low_conf_rate": _safe_float(
            out.get("max_low_conf_rate"), DEFAULT_THRESHOLDS["max_low_conf_rate"]
        ),
        "low_conf_threshold": _safe_float(
            out.get("low_conf_threshold"), DEFAULT_THRESHOLDS["low_conf_threshold"]
        ),
        "require_strip_text_entities": _safe_bool(
            out.get("require_strip_text_entities"),
            DEFAULT_THRESHOLDS["require_strip_text_entities"],
        ),
        "require_mask_filename": _safe_bool(
            out.get("require_mask_filename"),
            DEFAULT_THRESHOLDS["require_mask_filename"],
        ),
    }


def evaluate_blind_gate(summary: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []

    eval_opts = summary.get("eval_options") if isinstance(summary.get("eval_options"), dict) else {}
    strip_text_entities = bool(eval_opts.get("strip_text_entities"))
    mask_filename = bool(eval_opts.get("mask_filename"))

    if thresholds["require_strip_text_entities"] and not strip_text_entities:
        failures.append("eval_options.strip_text_entities must be true for blind gate.")
    if thresholds["require_mask_filename"] and not mask_filename:
        failures.append("eval_options.mask_filename must be true for blind gate.")

    status_counts = (
        summary.get("status_counts")
        if isinstance(summary.get("status_counts"), dict)
        else {}
    )
    ok_count = _safe_int(status_counts.get("ok"), 0)
    sampled_files = _safe_int(summary.get("sampled_files"), 0)
    denominator = ok_count if ok_count > 0 else sampled_files

    accuracy_val = summary.get("accuracy")
    accuracy: float | None
    if accuracy_val is None:
        accuracy = None
        failures.append("summary.accuracy missing; provide manifest labels for blind gate.")
    else:
        accuracy = _safe_float(accuracy_val, 0.0)
        if accuracy < thresholds["min_accuracy"]:
            failures.append(
                f"accuracy {accuracy:.4f} < min_accuracy {thresholds['min_accuracy']:.4f}"
            )

    distinct_count = 0
    pred_labels = summary.get("pred_labels") if isinstance(summary.get("pred_labels"), dict) else {}
    if pred_labels:
        distinct_count = _safe_int(pred_labels.get("distinct_canon_count"), 0)
        if distinct_count <= 0:
            distinct_count = _safe_int(pred_labels.get("distinct_count"), 0)
    if distinct_count <= 0:
        top_pred = summary.get("top_pred_labels_canon") or summary.get("top_pred_labels") or []
        if isinstance(top_pred, list):
            distinct_count = len(top_pred)
    if distinct_count < thresholds["min_distinct_pred_labels"]:
        failures.append(
            "distinct predicted labels "
            f"{distinct_count} < min_distinct_pred_labels "
            f"{thresholds['min_distinct_pred_labels']}"
        )

    top_pred = summary.get("top_pred_labels_canon") or summary.get("top_pred_labels") or []
    top_pred_count = 0
    top_pred_label = ""
    if isinstance(top_pred, list) and top_pred:
        first = top_pred[0]
        if isinstance(first, list) and len(first) >= 2:
            top_pred_label = str(first[0])
            top_pred_count = _safe_int(first[1], 0)
        elif isinstance(first, tuple) and len(first) >= 2:
            top_pred_label = str(first[0])
            top_pred_count = _safe_int(first[1], 0)
    top_pred_ratio = (float(top_pred_count) / float(denominator)) if denominator > 0 else 0.0
    if denominator <= 0:
        warnings.append("Cannot compute top_pred_ratio: ok/sample count is zero.")
    elif top_pred_ratio > thresholds["max_top_pred_ratio"]:
        failures.append(
            f"top_pred_ratio {top_pred_ratio:.4f} > max_top_pred_ratio "
            f"{thresholds['max_top_pred_ratio']:.4f} (label={top_pred_label})"
        )

    confidence = summary.get("confidence") if isinstance(summary.get("confidence"), dict) else {}
    low_conf_rate = _safe_float(confidence.get("low_conf_rate"), 0.0)
    summary_low_conf_thr = _safe_float(
        confidence.get("low_conf_threshold"), thresholds["low_conf_threshold"]
    )
    if abs(summary_low_conf_thr - thresholds["low_conf_threshold"]) > 1e-9:
        warnings.append(
            "Summary low_conf_threshold "
            f"{summary_low_conf_thr:.4f} != gate threshold "
            f"{thresholds['low_conf_threshold']:.4f}"
        )
    if low_conf_rate > thresholds["max_low_conf_rate"]:
        failures.append(
            f"low_conf_rate {low_conf_rate:.4f} > max_low_conf_rate "
            f"{thresholds['max_low_conf_rate']:.4f}"
        )

    status = "passed" if not failures else "failed"
    return {
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "metrics": {
            "sampled_files": sampled_files,
            "ok_count": ok_count,
            "accuracy": accuracy,
            "distinct_pred_labels": distinct_count,
            "top_pred_label": top_pred_label,
            "top_pred_count": top_pred_count,
            "top_pred_ratio": round(top_pred_ratio, 6),
            "low_conf_rate": round(low_conf_rate, 6),
            "low_conf_threshold": round(summary_low_conf_thr, 6),
            "eval_options": {
                "strip_text_entities": strip_text_entities,
                "mask_filename": mask_filename,
            },
        },
        "thresholds": thresholds,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Graph2D blind evaluation gate.")
    parser.add_argument("--summary-json", required=True, help="Path to blind summary JSON.")
    parser.add_argument(
        "--config",
        default="config/graph2d_blind_gate.yaml",
        help="Optional YAML config path.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional report JSON output path.",
    )
    parser.add_argument("--min-accuracy", type=float, default=None)
    parser.add_argument("--max-top-pred-ratio", type=float, default=None)
    parser.add_argument("--min-distinct-pred-labels", type=int, default=None)
    parser.add_argument("--max-low-conf-rate", type=float, default=None)
    parser.add_argument("--low-conf-threshold", type=float, default=None)
    parser.add_argument("--require-strip-text-entities", type=str, default=None)
    parser.add_argument("--require-mask-filename", type=str, default=None)
    args = parser.parse_args(argv)

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        raise SystemExit(f"summary not found: {summary_path}")

    config_payload = _load_yaml_defaults(args.config, "graph2d_blind_gate")
    thresholds = _resolve_thresholds(
        config_payload=config_payload,
        cli_overrides={
            "min_accuracy": args.min_accuracy,
            "max_top_pred_ratio": args.max_top_pred_ratio,
            "min_distinct_pred_labels": args.min_distinct_pred_labels,
            "max_low_conf_rate": args.max_low_conf_rate,
            "low_conf_threshold": args.low_conf_threshold,
            "require_strip_text_entities": args.require_strip_text_entities,
            "require_mask_filename": args.require_mask_filename,
        },
    )
    summary = _read_json(summary_path)
    report = evaluate_blind_gate(summary=summary, thresholds=thresholds)

    output_path = Path(args.output) if args.output else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
