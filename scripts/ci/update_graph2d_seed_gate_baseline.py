#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _safe_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_of(value: Any) -> str:
    text = _canonical_json(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _context_payload(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "config": _safe_str(summary.get("config"), ""),
        "training_profile": _safe_str(summary.get("training_profile"), ""),
        "manifest_label_mode": _safe_str(summary.get("manifest_label_mode"), ""),
        "seeds": _safe_int_list(summary.get("seeds")),
        "num_runs": _safe_int(summary.get("num_runs"), 0),
        "max_samples": _safe_int(summary.get("max_samples"), 0),
        "min_label_confidence": _safe_float(summary.get("min_label_confidence"), 0.0),
        "force_normalize_labels": _safe_str(
            summary.get("force_normalize_labels"), "auto"
        ),
        "force_clean_min_count": _safe_int(summary.get("force_clean_min_count"), -1),
        "strict_low_conf_threshold": _safe_float(
            summary.get("strict_low_conf_threshold"), 0.2
        ),
    }


def _channel_payload(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "strict_accuracy_mean": _safe_float(summary.get("strict_accuracy_mean"), 0.0),
        "strict_accuracy_min": _safe_float(summary.get("strict_accuracy_min"), 0.0),
        "strict_accuracy_max": _safe_float(summary.get("strict_accuracy_max"), 0.0),
        "strict_top_pred_ratio_mean": _safe_float(
            summary.get("strict_top_pred_ratio_mean"), 0.0
        ),
        "strict_top_pred_ratio_max": _safe_float(
            summary.get("strict_top_pred_ratio_max"), 0.0
        ),
        "strict_low_conf_threshold": _safe_float(
            summary.get("strict_low_conf_threshold"), 0.2
        ),
        "strict_low_conf_ratio_mean": _safe_float(
            summary.get("strict_low_conf_ratio_mean"), 0.0
        ),
        "strict_low_conf_ratio_max": _safe_float(
            summary.get("strict_low_conf_ratio_max"), 0.0
        ),
        "manifest_distinct_labels_min": _safe_int(
            summary.get("manifest_distinct_labels_min"), 0
        ),
        "manifest_distinct_labels_max": _safe_int(
            summary.get("manifest_distinct_labels_max"), 0
        ),
        "gate_passed": bool(
            (summary.get("gate") if isinstance(summary.get("gate"), dict) else {}).get(
                "passed", False
            )
        ),
        "context": _context_payload(summary),
    }


def build_baseline(
    *,
    standard_summary: Dict[str, Any],
    strict_summary: Dict[str, Any],
    standard_summary_path: str,
    strict_summary_path: str,
    snapshot_ref: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "date": dt.date.today().isoformat(),
        "source": {
            "standard_summary_json": str(standard_summary_path),
            "strict_summary_json": str(strict_summary_path),
            "snapshot_ref": str(snapshot_ref),
        },
        "standard": _channel_payload(standard_summary),
        "strict": _channel_payload(strict_summary),
    }
    payload["integrity"] = {
        "algorithm": "sha256-canonical-json",
        "standard_channel_sha256": _sha256_of(payload["standard"]),
        "strict_channel_sha256": _sha256_of(payload["strict"]),
        "payload_core_sha256": _sha256_of(
            {
                "date": payload["date"],
                "source": payload["source"],
                "standard": payload["standard"],
                "strict": payload["strict"],
            }
        ),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update Graph2D seed gate stable baseline json from latest summaries."
    )
    parser.add_argument(
        "--standard-summary-json",
        default="/tmp/graph2d-seed-gate/seed_sweep_summary.json",
        help="Standard channel summary json path.",
    )
    parser.add_argument(
        "--strict-summary-json",
        default="/tmp/graph2d-seed-gate-strict/seed_sweep_summary.json",
        help="Strict channel summary json path.",
    )
    parser.add_argument(
        "--output-baseline-json",
        default="config/graph2d_seed_gate_baseline.json",
        help="Stable baseline output path.",
    )
    parser.add_argument(
        "--snapshot-output-json",
        default="",
        help=(
            "Optional dated snapshot output path. "
            "Default: reports/experiments/<YYYYMMDD>/graph2d_seed_gate_baseline_snapshot_<YYYYMMDD>.json"
        ),
    )
    args = parser.parse_args()

    standard_summary_path = Path(args.standard_summary_json)
    strict_summary_path = Path(args.strict_summary_json)
    standard_summary = _read_json(standard_summary_path)
    strict_summary = _read_json(strict_summary_path)
    if not standard_summary:
        print(f"Missing/invalid standard summary: {standard_summary_path}")
        return 2
    if not strict_summary:
        print(f"Missing/invalid strict summary: {strict_summary_path}")
        return 2

    stamp = dt.date.today().strftime("%Y%m%d")
    default_snapshot_path = (
        Path("reports")
        / "experiments"
        / stamp
        / f"graph2d_seed_gate_baseline_snapshot_{stamp}.json"
    )
    snapshot_output = (
        Path(args.snapshot_output_json) if str(args.snapshot_output_json).strip() else default_snapshot_path
    )
    baseline_output = Path(args.output_baseline_json)

    baseline_payload = build_baseline(
        standard_summary=standard_summary,
        strict_summary=strict_summary,
        standard_summary_path=str(standard_summary_path),
        strict_summary_path=str(strict_summary_path),
        snapshot_ref=str(snapshot_output),
    )

    snapshot_output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_output.write_text(
        json.dumps(baseline_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    baseline_output.parent.mkdir(parents=True, exist_ok=True)
    baseline_output.write_text(
        json.dumps(baseline_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"baseline_json={baseline_output}")
    print(f"snapshot_json={snapshot_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
