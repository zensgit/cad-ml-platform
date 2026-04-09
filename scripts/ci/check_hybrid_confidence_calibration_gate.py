#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "min_samples": 30,
    "max_ece_increase": 0.02,
    "max_brier_increase": 0.02,
    "max_mce_increase": 0.03,
    "max_ece_vs_before_increase": 0.00,
    "max_brier_vs_before_increase": 0.00,
    "max_mce_vs_before_increase": 0.00,
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


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_yaml(path: Path) -> Dict[str, Any]:
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
    return payload if isinstance(payload, dict) else {}


def _extract_metrics(payload: Dict[str, Any], key: str) -> Dict[str, float]:
    block = payload.get(key)
    if not isinstance(block, dict):
        return {}
    return {
        "ece": _safe_float(block.get("ece"), 0.0),
        "brier_score": _safe_float(block.get("brier_score"), 0.0),
        "mce": _safe_float(block.get("mce"), 0.0),
    }


def _resolve_thresholds(config_payload: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(DEFAULT_THRESHOLDS)
    section = config_payload.get("hybrid_confidence_calibration_gate")
    if isinstance(section, dict):
        for key in DEFAULT_THRESHOLDS.keys():
            if key in section:
                merged[key] = section.get(key)
    return {
        "min_samples": max(1, _safe_int(merged.get("min_samples"), 30)),
        "max_ece_increase": _safe_float(merged.get("max_ece_increase"), 0.02),
        "max_brier_increase": _safe_float(merged.get("max_brier_increase"), 0.02),
        "max_mce_increase": _safe_float(merged.get("max_mce_increase"), 0.03),
        "max_ece_vs_before_increase": _safe_float(
            merged.get("max_ece_vs_before_increase"), 0.0
        ),
        "max_brier_vs_before_increase": _safe_float(
            merged.get("max_brier_vs_before_increase"), 0.0
        ),
        "max_mce_vs_before_increase": _safe_float(
            merged.get("max_mce_vs_before_increase"), 0.0
        ),
    }


def evaluate_gate(
    *,
    current: Dict[str, Any],
    baseline: Dict[str, Any],
    thresholds: Dict[str, Any],
    missing_mode: str,
) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []

    if not current:
        status = "failed" if missing_mode == "fail" else "skipped"
        reason = "current_missing"
        return {
            "status": status,
            "reason": reason,
            "failures": failures
            + (["Current calibration report missing"] if status == "failed" else []),
            "warnings": warnings
            + (["Current calibration report missing"] if status != "failed" else []),
            "thresholds": thresholds,
            "current": {},
            "baseline": {},
        }

    n_samples = _safe_int(current.get("n_samples"), 0)
    if n_samples < _safe_int(thresholds.get("min_samples"), 30):
        failures.append(
            f"n_samples_too_low:{n_samples} < {int(thresholds.get('min_samples', 30))}"
        )

    current_before = _extract_metrics(current, "metrics_before")
    current_after = _extract_metrics(current, "metrics_after")
    if not current_after:
        failures.append("current_metrics_after_missing")

    baseline_after = _extract_metrics(baseline, "metrics_after") if baseline else {}
    if not baseline_after:
        warnings.append("baseline_metrics_after_missing")

    # Current vs baseline regression.
    if baseline_after and current_after:
        delta_ece = current_after["ece"] - baseline_after["ece"]
        delta_brier = current_after["brier_score"] - baseline_after["brier_score"]
        delta_mce = current_after["mce"] - baseline_after["mce"]
        if delta_ece > _safe_float(thresholds.get("max_ece_increase"), 0.02):
            failures.append(
                f"ece_regression:{delta_ece:.6f} > {float(thresholds.get('max_ece_increase', 0.02)):.6f}"
            )
        if delta_brier > _safe_float(thresholds.get("max_brier_increase"), 0.02):
            failures.append(
                f"brier_regression:{delta_brier:.6f} > {float(thresholds.get('max_brier_increase', 0.02)):.6f}"
            )
        if delta_mce > _safe_float(thresholds.get("max_mce_increase"), 0.03):
            failures.append(
                f"mce_regression:{delta_mce:.6f} > {float(thresholds.get('max_mce_increase', 0.03)):.6f}"
            )

    # Current after should not be worse than current before.
    if current_before and current_after:
        delta_ece_before = current_after["ece"] - current_before["ece"]
        delta_brier_before = (
            current_after["brier_score"] - current_before["brier_score"]
        )
        delta_mce_before = current_after["mce"] - current_before["mce"]
        if delta_ece_before > _safe_float(
            thresholds.get("max_ece_vs_before_increase"), 0.0
        ):
            failures.append(
                f"ece_vs_before_regression:{delta_ece_before:.6f} > "
                f"{float(thresholds.get('max_ece_vs_before_increase', 0.0)):.6f}"
            )
        if delta_brier_before > _safe_float(
            thresholds.get("max_brier_vs_before_increase"), 0.0
        ):
            failures.append(
                f"brier_vs_before_regression:{delta_brier_before:.6f} > "
                f"{float(thresholds.get('max_brier_vs_before_increase', 0.0)):.6f}"
            )
        if delta_mce_before > _safe_float(
            thresholds.get("max_mce_vs_before_increase"), 0.0
        ):
            failures.append(
                f"mce_vs_before_regression:{delta_mce_before:.6f} > "
                f"{float(thresholds.get('max_mce_vs_before_increase', 0.0)):.6f}"
            )

    status = "failed" if failures else "passed"
    return {
        "status": status,
        "reason": "ok" if status == "passed" else "threshold_violation",
        "failures": failures,
        "warnings": warnings,
        "thresholds": thresholds,
        "current": {
            "n_samples": n_samples,
            "metrics_before": current_before,
            "metrics_after": current_after,
            "path": str(current.get("_path") or ""),
        },
        "baseline": {
            "metrics_after": baseline_after,
            "path": str(baseline.get("_path") or "") if baseline else "",
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check Hybrid confidence calibration gate regression."
    )
    parser.add_argument(
        "--current-json",
        default="reports/calibration/hybrid_confidence_calibration_latest.json",
        help="Current calibration report json path.",
    )
    parser.add_argument(
        "--baseline-json",
        default="config/hybrid_confidence_calibration_baseline.json",
        help="Baseline calibration report json path.",
    )
    parser.add_argument(
        "--config",
        default="config/hybrid_confidence_calibration_gate.yaml",
        help="Gate threshold config yaml.",
    )
    parser.add_argument(
        "--missing-mode",
        choices=["skip", "fail"],
        default="skip",
        help="Behavior when current report is missing.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output path for gate report json.",
    )
    args = parser.parse_args(argv)

    current_path = Path(args.current_json).expanduser()
    baseline_path = Path(args.baseline_json).expanduser()
    config_path = Path(args.config).expanduser()

    current = _read_json(current_path)
    if current:
        current["_path"] = str(current_path)
    baseline = _read_json(baseline_path)
    if baseline:
        baseline["_path"] = str(baseline_path)

    config_payload = _load_yaml(config_path)
    thresholds = _resolve_thresholds(config_payload)
    report = evaluate_gate(
        current=current,
        baseline=baseline,
        thresholds=thresholds,
        missing_mode=str(args.missing_mode),
    )
    report.update(
        {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config": str(config_path),
            "current_json": str(current_path),
            "baseline_json": str(baseline_path),
            "missing_mode": str(args.missing_mode),
        }
    )

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"status={report['status']}")
    print(f"reason={report['reason']}")
    print(f"failures={len(report.get('failures') or [])}")
    print(f"warnings={len(report.get('warnings') or [])}")

    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
