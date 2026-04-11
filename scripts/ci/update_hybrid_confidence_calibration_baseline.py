#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


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


def _safe_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_of(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _calibration_payload(current: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": _safe_str(current.get("status"), "unknown"),
        "method": _safe_str(current.get("method"), ""),
        "per_source": bool(current.get("per_source", False)),
        "effective_per_source": bool(current.get("effective_per_source", False)),
        "n_rows": _safe_int(current.get("n_rows"), 0),
        "n_samples": _safe_int(current.get("n_samples"), 0),
        "min_samples": _safe_int(current.get("min_samples"), 0),
        "min_samples_per_source": _safe_int(current.get("min_samples_per_source"), 0),
        "dropped_bad_confidence": _safe_int(current.get("dropped_bad_confidence"), 0),
        "dropped_no_correctness": _safe_int(current.get("dropped_no_correctness"), 0),
        "pair_counts": _safe_dict(current.get("pair_counts")),
        "source_counts": _safe_dict(current.get("source_counts")),
        "temperature": _safe_float(current.get("temperature"), 0.0),
        "source_temperatures": _safe_dict(current.get("source_temperatures")),
        "metrics_before": _safe_dict(current.get("metrics_before")),
        "metrics_after": _safe_dict(current.get("metrics_after")),
    }


def build_baseline(
    *,
    current: Dict[str, Any],
    current_path: str,
    snapshot_ref: str,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "date": dt.date.today().isoformat(),
        "generated_at": generated_at
        or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "current_report_json": str(current_path),
            "snapshot_ref": str(snapshot_ref),
        },
        "calibration": _calibration_payload(current),
    }
    payload["integrity"] = {
        "algorithm": "sha256-canonical-json",
        "calibration_sha256": _sha256_of(payload["calibration"]),
        "payload_core_sha256": _sha256_of(
            {
                "date": payload["date"],
                "generated_at": payload["generated_at"],
                "source": payload["source"],
                "calibration": payload["calibration"],
            }
        ),
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Update Hybrid confidence calibration baseline json."
    )
    parser.add_argument(
        "--current-json",
        default="reports/calibration/hybrid_confidence_calibration_latest.json",
        help="Current calibration report json path.",
    )
    parser.add_argument(
        "--output-baseline-json",
        default="config/hybrid_confidence_calibration_baseline.json",
        help="Stable baseline output path.",
    )
    parser.add_argument(
        "--snapshot-output-json",
        default="",
        help=(
            "Optional dated snapshot output path. "
            "Default: reports/experiments/<YYYYMMDD>/"
            "hybrid_confidence_calibration_baseline_snapshot_<YYYYMMDD>.json"
        ),
    )
    parser.add_argument(
        "--allow-non-ok-status",
        action="store_true",
        help="Allow writing baseline when current status is not 'ok'.",
    )
    args = parser.parse_args(argv)

    current_path = Path(args.current_json).expanduser()
    current = _read_json(current_path)
    if not current:
        print(f"Missing/invalid current report: {current_path}")
        return 2

    status = _safe_str(current.get("status"), "unknown")
    if status != "ok" and not args.allow_non_ok_status:
        print(
            "Current calibration report status is not ok "
            f"(status={status!r}); use --allow-non-ok-status to override."
        )
        return 2

    stamp = dt.date.today().strftime("%Y%m%d")
    default_snapshot_path = (
        Path("reports")
        / "experiments"
        / stamp
        / f"hybrid_confidence_calibration_baseline_snapshot_{stamp}.json"
    )
    snapshot_output = (
        Path(args.snapshot_output_json).expanduser()
        if str(args.snapshot_output_json).strip()
        else default_snapshot_path
    )
    baseline_output = Path(args.output_baseline_json).expanduser()

    baseline_payload = build_baseline(
        current=current,
        current_path=str(current_path),
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
