#!/usr/bin/env python3
"""Gate "surpass benchmark" targets for hybrid blind + calibration outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "min_hybrid_accuracy": 0.60,
    "min_hybrid_gain_vs_graph2d": 0.00,
    "max_calibration_ece": 0.08,
    "missing_mode": "skip",
    "require_real_blind_dataset": True,
    "allowed_blind_dataset_sources": ["configured_dxf_dir"],
}


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _normalize_source_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _read_json_object(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _load_yaml_defaults(config_path: str, section: str) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists() or not path.is_file():
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


def _resolve_missing_mode(value: Any, default: str = "skip") -> str:
    token = str(value or default).strip().lower()
    if token in {"skip", "fail"}:
        return token
    return str(default).strip().lower() or "skip"


def _resolve_thresholds(
    config_payload: Dict[str, Any],
    cli_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(DEFAULT_THRESHOLDS)
    merged.update(config_payload)
    for key, value in cli_overrides.items():
        if value is not None:
            merged[key] = value
    return {
        "min_hybrid_accuracy": float(
            _optional_float(merged.get("min_hybrid_accuracy"))
            or DEFAULT_THRESHOLDS["min_hybrid_accuracy"]
        ),
        "min_hybrid_gain_vs_graph2d": float(
            _optional_float(merged.get("min_hybrid_gain_vs_graph2d"))
            or DEFAULT_THRESHOLDS["min_hybrid_gain_vs_graph2d"]
        ),
        "max_calibration_ece": float(
            _optional_float(merged.get("max_calibration_ece"))
            or DEFAULT_THRESHOLDS["max_calibration_ece"]
        ),
        "missing_mode": _resolve_missing_mode(
            merged.get("missing_mode"), str(DEFAULT_THRESHOLDS["missing_mode"])
        ),
        "require_real_blind_dataset": _coerce_bool(
            merged.get("require_real_blind_dataset"),
            bool(DEFAULT_THRESHOLDS["require_real_blind_dataset"]),
        ),
        "allowed_blind_dataset_sources": _normalize_source_list(
            merged.get("allowed_blind_dataset_sources")
            or DEFAULT_THRESHOLDS["allowed_blind_dataset_sources"]
        ),
    }


def _build_check(
    *,
    name: str,
    actual: Optional[float],
    threshold: float,
    comparator: str,
    passed: bool,
    source: str,
    skipped: bool,
    message: str,
) -> Dict[str, Any]:
    return {
        "name": name,
        "actual": actual,
        "threshold": threshold,
        "comparator": comparator,
        "passed": bool(passed),
        "source": source,
        "skipped": bool(skipped),
        "message": str(message),
    }


def _evaluate_ge(actual: Optional[float], threshold: float) -> bool:
    if actual is None:
        return False
    return float(actual) >= float(threshold)


def _evaluate_le(actual: Optional[float], threshold: float) -> bool:
    if actual is None:
        return False
    return float(actual) <= float(threshold)


def evaluate_superpass_targets(
    *,
    hybrid_blind_gate_report: Optional[Dict[str, Any]],
    hybrid_calibration_json: Optional[Dict[str, Any]],
    thresholds: Dict[str, Any],
    missing_mode: str,
    hybrid_blind_dataset_source: Optional[str] = None,
) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []
    checks: List[Dict[str, Any]] = []
    mode = _resolve_missing_mode(missing_mode, str(thresholds.get("missing_mode", "skip")))

    gate_metrics = (
        hybrid_blind_gate_report.get("metrics", {})
        if isinstance(hybrid_blind_gate_report, dict)
        and isinstance(hybrid_blind_gate_report.get("metrics"), dict)
        else {}
    )
    if not gate_metrics:
        message = "hybrid blind gate report missing or invalid."
        if mode == "fail":
            failures.append(message)
        else:
            warnings.append(message)

    blind_inputs: Dict[str, Any] = {}
    if isinstance(hybrid_blind_gate_report, dict):
        if isinstance(hybrid_blind_gate_report.get("input_summary"), dict):
            blind_inputs = hybrid_blind_gate_report.get("input_summary", {})
        elif isinstance(hybrid_blind_gate_report.get("inputs"), dict):
            blind_inputs = hybrid_blind_gate_report.get("inputs", {})
    blind_dataset_source = (
        _optional_str(hybrid_blind_dataset_source)
        or _optional_str(blind_inputs.get("dataset_source"))
        or _optional_str(
            hybrid_blind_gate_report.get("dataset_source")
            if isinstance(hybrid_blind_gate_report, dict)
            else None
        )
    )
    require_real_blind_dataset = bool(thresholds["require_real_blind_dataset"])
    allowed_blind_dataset_sources = list(thresholds["allowed_blind_dataset_sources"])
    blind_dataset_qualified = (
        (not require_real_blind_dataset)
        or (
            blind_dataset_source is None
            or blind_dataset_source in allowed_blind_dataset_sources
        )
    )
    unsupported_source_message = ""
    if require_real_blind_dataset and blind_dataset_source is not None and not blind_dataset_qualified:
        allowed_text = ", ".join(allowed_blind_dataset_sources) or "configured_dxf_dir"
        source_text = blind_dataset_source
        unsupported_source_message = (
            f"hybrid blind dataset_source {source_text!r} is advisory only for "
            f"superpass targets; strict blind targets require one of: {allowed_text}."
        )
        warnings.append(unsupported_source_message)

    hybrid_accuracy = _optional_float(gate_metrics.get("hybrid_accuracy"))
    min_hybrid_accuracy = float(thresholds["min_hybrid_accuracy"])
    if unsupported_source_message:
        checks.append(
            _build_check(
                name="hybrid_accuracy",
                actual=hybrid_accuracy,
                threshold=min_hybrid_accuracy,
                comparator=">=",
                passed=True,
                source="hybrid_blind_gate",
                skipped=True,
                message=unsupported_source_message,
            )
        )
    elif hybrid_accuracy is None:
        message = "hybrid_accuracy unavailable."
        if mode == "fail":
            failures.append(message)
        else:
            warnings.append(message)
        checks.append(
            _build_check(
                name="hybrid_accuracy",
                actual=None,
                threshold=min_hybrid_accuracy,
                comparator=">=",
                passed=(mode != "fail"),
                source="hybrid_blind_gate",
                skipped=(mode != "fail"),
                message=message,
            )
        )
    else:
        passed = _evaluate_ge(hybrid_accuracy, min_hybrid_accuracy)
        if not passed:
            failures.append(
                "hybrid_accuracy {:.6f} < min_hybrid_accuracy {:.6f}".format(
                    hybrid_accuracy, min_hybrid_accuracy
                )
            )
        checks.append(
            _build_check(
                name="hybrid_accuracy",
                actual=hybrid_accuracy,
                threshold=min_hybrid_accuracy,
                comparator=">=",
                passed=passed,
                source="hybrid_blind_gate",
                skipped=False,
                message=(
                    "ok"
                    if passed
                    else "hybrid_accuracy {:.6f} < {:.6f}".format(
                        hybrid_accuracy, min_hybrid_accuracy
                    )
                ),
            )
        )

    hybrid_gain = _optional_float(gate_metrics.get("hybrid_gain_vs_graph2d"))
    min_hybrid_gain = float(thresholds["min_hybrid_gain_vs_graph2d"])
    if unsupported_source_message:
        checks.append(
            _build_check(
                name="hybrid_gain_vs_graph2d",
                actual=hybrid_gain,
                threshold=min_hybrid_gain,
                comparator=">=",
                passed=True,
                source="hybrid_blind_gate",
                skipped=True,
                message=unsupported_source_message,
            )
        )
    elif hybrid_gain is None:
        message = "hybrid_gain_vs_graph2d unavailable."
        if mode == "fail":
            failures.append(message)
        else:
            warnings.append(message)
        checks.append(
            _build_check(
                name="hybrid_gain_vs_graph2d",
                actual=None,
                threshold=min_hybrid_gain,
                comparator=">=",
                passed=(mode != "fail"),
                source="hybrid_blind_gate",
                skipped=(mode != "fail"),
                message=message,
            )
        )
    else:
        passed = _evaluate_ge(hybrid_gain, min_hybrid_gain)
        if not passed:
            failures.append(
                "hybrid_gain_vs_graph2d {:.6f} < min_hybrid_gain_vs_graph2d {:.6f}".format(
                    hybrid_gain, min_hybrid_gain
                )
            )
        checks.append(
            _build_check(
                name="hybrid_gain_vs_graph2d",
                actual=hybrid_gain,
                threshold=min_hybrid_gain,
                comparator=">=",
                passed=passed,
                source="hybrid_blind_gate",
                skipped=False,
                message=(
                    "ok"
                    if passed
                    else "hybrid_gain_vs_graph2d {:.6f} < {:.6f}".format(
                        hybrid_gain, min_hybrid_gain
                    )
                ),
            )
        )

    calibration_metrics_after = (
        hybrid_calibration_json.get("metrics_after", {})
        if isinstance(hybrid_calibration_json, dict)
        and isinstance(hybrid_calibration_json.get("metrics_after"), dict)
        else {}
    )
    if not calibration_metrics_after:
        message = "hybrid calibration metrics_after missing or invalid."
        if mode == "fail":
            failures.append(message)
        else:
            warnings.append(message)

    calibration_ece = _optional_float(calibration_metrics_after.get("ece"))
    max_calibration_ece = float(thresholds["max_calibration_ece"])
    if calibration_ece is None:
        message = "calibration_ece unavailable."
        if mode == "fail":
            failures.append(message)
        else:
            warnings.append(message)
        checks.append(
            _build_check(
                name="calibration_ece",
                actual=None,
                threshold=max_calibration_ece,
                comparator="<=",
                passed=(mode != "fail"),
                source="hybrid_calibration",
                skipped=(mode != "fail"),
                message=message,
            )
        )
    else:
        passed = _evaluate_le(calibration_ece, max_calibration_ece)
        if not passed:
            failures.append(
                "calibration_ece {:.6f} > max_calibration_ece {:.6f}".format(
                    calibration_ece, max_calibration_ece
                )
            )
        checks.append(
            _build_check(
                name="calibration_ece",
                actual=calibration_ece,
                threshold=max_calibration_ece,
                comparator="<=",
                passed=passed,
                source="hybrid_calibration",
                skipped=False,
                message=(
                    "ok"
                    if passed
                    else "calibration_ece {:.6f} > {:.6f}".format(
                        calibration_ece, max_calibration_ece
                    )
                ),
            )
        )

    status = "passed" if not failures else "failed"
    return {
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "checks": checks,
        "thresholds": {
            "min_hybrid_accuracy": min_hybrid_accuracy,
            "min_hybrid_gain_vs_graph2d": min_hybrid_gain,
            "max_calibration_ece": max_calibration_ece,
            "missing_mode": mode,
            "require_real_blind_dataset": require_real_blind_dataset,
            "allowed_blind_dataset_sources": allowed_blind_dataset_sources,
        },
        "inputs": {
            "hybrid_blind_gate_report_present": bool(gate_metrics),
            "hybrid_calibration_present": bool(calibration_metrics_after),
            "hybrid_blind_dataset_source": blind_dataset_source or "",
            "hybrid_blind_dataset_qualified": bool(blind_dataset_qualified),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check hybrid superpass targets from hybrid blind gate + "
            "hybrid calibration artifacts."
        )
    )
    parser.add_argument(
        "--hybrid-blind-gate-report",
        default="reports/history_sequence_eval/hybrid_blind_gate_report.json",
        help="Path to hybrid blind gate report JSON.",
    )
    parser.add_argument(
        "--hybrid-calibration-json",
        default="models/calibration/hybrid_confidence_calibration.json",
        help="Path to hybrid calibration output JSON.",
    )
    parser.add_argument(
        "--config",
        default="config/hybrid_superpass_targets.yaml",
        help="Optional YAML config path.",
    )
    parser.add_argument(
        "--missing-mode",
        default=None,
        choices=["skip", "fail"],
        help="How to handle missing inputs/metrics.",
    )
    parser.add_argument(
        "--hybrid-blind-dataset-source",
        default=None,
        help="Optional blind benchmark dataset source label (for example configured_dxf_dir or synthetic_manifest).",
    )
    parser.add_argument("--min-hybrid-accuracy", type=float, default=None)
    parser.add_argument("--min-hybrid-gain-vs-graph2d", type=float, default=None)
    parser.add_argument("--max-calibration-ece", type=float, default=None)
    parser.add_argument("--output", default="", help="Optional report output JSON path.")
    args = parser.parse_args(argv)

    config_payload = _load_yaml_defaults(str(args.config), "hybrid_superpass")
    thresholds = _resolve_thresholds(
        config_payload=config_payload,
        cli_overrides={
            "missing_mode": args.missing_mode,
            "min_hybrid_accuracy": args.min_hybrid_accuracy,
            "min_hybrid_gain_vs_graph2d": args.min_hybrid_gain_vs_graph2d,
            "max_calibration_ece": args.max_calibration_ece,
        },
    )
    missing_mode = _resolve_missing_mode(
        args.missing_mode or thresholds.get("missing_mode"),
        str(DEFAULT_THRESHOLDS["missing_mode"]),
    )

    gate_report = _read_json_object(Path(args.hybrid_blind_gate_report).expanduser())
    calibration_json = _read_json_object(Path(args.hybrid_calibration_json).expanduser())

    report = evaluate_superpass_targets(
        hybrid_blind_gate_report=gate_report,
        hybrid_calibration_json=calibration_json,
        thresholds=thresholds,
        missing_mode=missing_mode,
        hybrid_blind_dataset_source=args.hybrid_blind_dataset_source,
    )

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            f"{json.dumps(report, ensure_ascii=False, indent=2)}\n",
            encoding="utf-8",
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
