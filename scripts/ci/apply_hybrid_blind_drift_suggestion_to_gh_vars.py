#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _safe_bool_text(value: Any) -> str:
    token = str(value).strip().lower()
    return "true" if token in {"1", "true", "yes", "on"} else "false"


def _safe_float_text(value: Any) -> str:
    try:
        return str(float(value))
    except Exception:
        return "0.0"


def _safe_int_text(value: Any) -> str:
    try:
        return str(int(value))
    except Exception:
        return "0"


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid json root: {path}")
    return payload


def _build_var_map(payload: Dict[str, Any]) -> Dict[str, str]:
    status = str(payload.get("status") or "").strip().lower()
    if status != "ok":
        raise ValueError(f"suggestion status must be 'ok', got: {status or 'unknown'}")

    thresholds = payload.get("recommended_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("missing recommended_thresholds in suggestion json")

    return {
        "HYBRID_BLIND_DRIFT_ALERT_MIN_REPORTS": _safe_int_text(
            thresholds.get("min_reports", 3)
        ),
        "HYBRID_BLIND_DRIFT_ALERT_MAX_ACC_DROP": _safe_float_text(
            thresholds.get("max_hybrid_accuracy_drop", 0.05)
        ),
        "HYBRID_BLIND_DRIFT_ALERT_MAX_GAIN_DROP": _safe_float_text(
            thresholds.get("max_gain_drop", 0.05)
        ),
        "HYBRID_BLIND_DRIFT_ALERT_MAX_COVERAGE_DROP": _safe_float_text(
            thresholds.get("max_coverage_drop", 0.10)
        ),
        "HYBRID_BLIND_DRIFT_ALERT_LABEL_SLICE_ENABLE": _safe_bool_text(
            thresholds.get("label_slice_enable", True)
        ),
        "HYBRID_BLIND_DRIFT_ALERT_LABEL_SLICE_MIN_COMMON": _safe_int_text(
            thresholds.get("label_slice_min_common", 2)
        ),
        "HYBRID_BLIND_DRIFT_ALERT_LABEL_SLICE_AUTO_CAP_MIN_COMMON": _safe_bool_text(
            thresholds.get("label_slice_auto_cap_min_common", True)
        ),
        "HYBRID_BLIND_DRIFT_ALERT_LABEL_SLICE_MIN_SUPPORT": _safe_int_text(
            thresholds.get("label_slice_min_support", 3)
        ),
        "HYBRID_BLIND_DRIFT_ALERT_LABEL_SLICE_MAX_ACC_DROP": _safe_float_text(
            thresholds.get("label_slice_max_hybrid_accuracy_drop", 0.15)
        ),
        "HYBRID_BLIND_DRIFT_ALERT_LABEL_SLICE_MAX_GAIN_DROP": _safe_float_text(
            thresholds.get("label_slice_max_gain_drop", 0.15)
        ),
        "HYBRID_BLIND_DRIFT_ALERT_FAMILY_SLICE_ENABLE": _safe_bool_text(
            thresholds.get("family_slice_enable", True)
        ),
        "HYBRID_BLIND_DRIFT_ALERT_FAMILY_SLICE_MIN_COMMON": _safe_int_text(
            thresholds.get("family_slice_min_common", 2)
        ),
        "HYBRID_BLIND_DRIFT_ALERT_FAMILY_SLICE_AUTO_CAP_MIN_COMMON": _safe_bool_text(
            thresholds.get("family_slice_auto_cap_min_common", True)
        ),
        "HYBRID_BLIND_DRIFT_ALERT_FAMILY_SLICE_MIN_SUPPORT": _safe_int_text(
            thresholds.get("family_slice_min_support", 5)
        ),
        "HYBRID_BLIND_DRIFT_ALERT_FAMILY_SLICE_MAX_ACC_DROP": _safe_float_text(
            thresholds.get("family_slice_max_hybrid_accuracy_drop", 0.20)
        ),
        "HYBRID_BLIND_DRIFT_ALERT_FAMILY_SLICE_MAX_GAIN_DROP": _safe_float_text(
            thresholds.get("family_slice_max_gain_drop", 0.20)
        ),
    }


def _print_plan(repo: str, var_map: Dict[str, str]) -> None:
    print("plan=gh_variables")
    print(f"repo={repo}")
    for key in sorted(var_map.keys()):
        print(f"{key}={var_map[key]}")


def _apply(repo: str, var_map: Dict[str, str]) -> List[Tuple[str, int, str]]:
    results: List[Tuple[str, int, str]] = []
    for key in sorted(var_map.keys()):
        value = var_map[key]
        proc = subprocess.run(
            ["gh", "variable", "set", key, "--repo", repo, "--body", value],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (proc.stdout or proc.stderr or "").strip()
        results.append((key, int(proc.returncode), out))
    return results


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply hybrid blind drift threshold suggestion JSON to GitHub variables."
    )
    parser.add_argument(
        "--suggestion-json",
        default="reports/eval_history/hybrid_blind_drift_threshold_suggestion.json",
    )
    parser.add_argument("--repo", required=True, help="GitHub repo, e.g. owner/repo")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    payload = _read_json(Path(args.suggestion_json))
    var_map = _build_var_map(payload)
    _print_plan(args.repo, var_map)

    if not args.apply:
        print("apply=false")
        return 0

    results = _apply(args.repo, var_map)
    failed = [row for row in results if row[1] != 0]
    for key, code, message in results:
        status = "ok" if code == 0 else "failed"
        print(f"result {key} status={status} code={code} message={message}")
    print(f"applied={len(results)} failed={len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
