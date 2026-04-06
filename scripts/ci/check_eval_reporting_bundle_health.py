#!/usr/bin/env python3
"""Health / freshness / pointer guard for the top-level eval reporting bundle."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_report_data_helpers import load_json_dict


def _file_age_hours(path: Path, *, now: Optional[datetime] = None) -> Optional[float]:
    if not path.exists():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    ref = now or datetime.now(timezone.utc)
    return max(0.0, (ref - mtime).total_seconds() / 3600.0)


def _check_artifact(
    name: str,
    path_text: str,
    *,
    max_age_hours: float,
    now: Optional[datetime] = None,
    expected_surface_kind: Optional[str] = None,
) -> Dict[str, Any]:
    if not path_text:
        return {"name": name, "status": "missing", "path": "", "detail": "path not set in bundle"}
    path = Path(path_text)
    if not path.exists():
        return {"name": name, "status": "missing", "path": path_text, "detail": "file does not exist"}
    age = _file_age_hours(path, now=now)
    if age is not None and age > max_age_hours:
        return {
            "name": name,
            "status": "stale",
            "path": path_text,
            "age_hours": round(age, 2),
            "max_age_hours": max_age_hours,
            "detail": f"age {age:.1f}h exceeds threshold {max_age_hours:.1f}h",
        }
    if expected_surface_kind is not None:
        payload = load_json_dict(path)
        actual_kind = str((payload or {}).get("surface_kind", "")).strip() if payload else ""
        if actual_kind and actual_kind != expected_surface_kind:
            return {
                "name": name,
                "status": "mismatch",
                "path": path_text,
                "expected_surface_kind": expected_surface_kind,
                "actual_surface_kind": actual_kind,
                "detail": f"expected surface_kind={expected_surface_kind}, got {actual_kind}",
            }
    return {"name": name, "status": "ok", "path": path_text}


def run_health_checks(
    eval_history_dir: Path,
    *,
    bundle_json_path: Optional[Path] = None,
    max_root_age_hours: float = 168.0,
    max_sub_bundle_age_hours: float = 168.0,
    max_report_age_hours: float = 336.0,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    bundle_path = bundle_json_path or (eval_history_dir / "eval_reporting_bundle.json")
    bundle = load_json_dict(bundle_path)

    checks: List[Dict[str, Any]] = []

    if bundle is None:
        checks.append({
            "name": "root_bundle",
            "status": "missing",
            "path": str(bundle_path),
            "detail": "top-level eval_reporting_bundle.json not found",
        })
        return _build_report(eval_history_dir, bundle_path, checks, now=now)

    # root bundle freshness
    root_check = _check_artifact(
        "root_bundle",
        str(bundle_path),
        max_age_hours=max_root_age_hours,
        now=now,
        expected_surface_kind="eval_reporting_bundle",
    )
    checks.append(root_check)

    # sub-bundles
    for name, key, kind in [
        ("eval_signal_bundle", "eval_signal_bundle_json", "eval_signal_reporting_bundle"),
        ("history_sequence_bundle", "history_sequence_bundle_json", "history_sequence_reporting_bundle"),
    ]:
        checks.append(_check_artifact(
            name,
            str(bundle.get(key) or ""),
            max_age_hours=max_sub_bundle_age_hours,
            now=now,
            expected_surface_kind=kind,
        ))

    # HTML reports
    for name, key in [
        ("static_report", "static_report_html"),
        ("interactive_report", "interactive_report_html"),
    ]:
        checks.append(_check_artifact(
            name,
            str(bundle.get(key) or ""),
            max_age_hours=max_report_age_hours,
            now=now,
        ))

    # plots dir
    plots_dir_text = str(bundle.get("plots_dir") or "")
    if plots_dir_text and not Path(plots_dir_text).is_dir():
        checks.append({
            "name": "plots_dir",
            "status": "missing",
            "path": plots_dir_text,
            "detail": "plots directory does not exist",
        })
    else:
        checks.append({"name": "plots_dir", "status": "ok", "path": plots_dir_text})

    return _build_report(eval_history_dir, bundle_path, checks, now=now)


def _build_report(
    eval_history_dir: Path,
    bundle_path: Path,
    checks: List[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    missing = [c for c in checks if c["status"] == "missing"]
    stale = [c for c in checks if c["status"] == "stale"]
    mismatch = [c for c in checks if c["status"] == "mismatch"]
    all_ok = not missing and not stale and not mismatch

    return {
        "status": "ok" if all_ok else "unhealthy",
        "surface_kind": "eval_reporting_bundle_health_report",
        "generated_at": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eval_history_dir": str(eval_history_dir),
        "bundle_json": str(bundle_path),
        "summary": {
            "ok": all_ok,
            "missing_count": len(missing),
            "stale_count": len(stale),
            "mismatch_count": len(mismatch),
        },
        "checks": checks,
        "missing_artifacts": [c["name"] for c in missing],
        "stale_artifacts": [c["name"] for c in stale],
        "mismatch_artifacts": [c["name"] for c in mismatch],
    }


def _build_health_markdown(report: Dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Eval Reporting Bundle Health Report",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Bundle JSON: `{report.get('bundle_json', '')}`",
        f"- Missing: `{summary.get('missing_count', 0)}`",
        f"- Stale: `{summary.get('stale_count', 0)}`",
        f"- Mismatch: `{summary.get('mismatch_count', 0)}`",
        "",
        "## Checks",
        "",
        "| Name | Status | Detail |",
        "|---|---|---|",
    ]
    for check in report.get("checks", []):
        lines.append(
            f"| {check.get('name', '')} | {check.get('status', '')} | {check.get('detail', '-')} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check health / freshness / pointer integrity of the top-level eval reporting bundle."
    )
    parser.add_argument(
        "--eval-history-dir",
        default="reports/eval_history",
        help="Directory containing eval-history artifacts.",
    )
    parser.add_argument(
        "--bundle-json",
        default="",
        help="Top-level bundle JSON path (default: <eval-history-dir>/eval_reporting_bundle.json).",
    )
    parser.add_argument(
        "--max-root-age-hours",
        type=float,
        default=168.0,
        help="Max age (hours) for root bundle before it is considered stale (default: 168 = 7 days).",
    )
    parser.add_argument(
        "--max-sub-bundle-age-hours",
        type=float,
        default=168.0,
        help="Max age (hours) for sub-bundles (default: 168).",
    )
    parser.add_argument(
        "--max-report-age-hours",
        type=float,
        default=336.0,
        help="Max age (hours) for HTML reports (default: 336 = 14 days).",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Health report JSON output path.",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Health report Markdown output path.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    eval_history_dir = Path(str(args.eval_history_dir))
    bundle_json_path = (
        Path(str(args.bundle_json))
        if str(args.bundle_json).strip()
        else None
    )
    output_json = (
        Path(str(args.output_json))
        if str(args.output_json).strip()
        else eval_history_dir / "eval_reporting_bundle_health_report.json"
    )
    output_md = (
        Path(str(args.output_md))
        if str(args.output_md).strip()
        else eval_history_dir / "eval_reporting_bundle_health_report.md"
    )

    report = run_health_checks(
        eval_history_dir,
        bundle_json_path=bundle_json_path,
        max_root_age_hours=args.max_root_age_hours,
        max_sub_bundle_age_hours=args.max_sub_bundle_age_hours,
        max_report_age_hours=args.max_report_age_hours,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_build_health_markdown(report), encoding="utf-8")

    print(f"Health report JSON: {output_json}")
    print(f"Health report Markdown: {output_md}")
    print(f"Status: {report['status']}")
    if report["missing_artifacts"]:
        print(f"Missing: {', '.join(report['missing_artifacts'])}")
    if report["stale_artifacts"]:
        print(f"Stale: {', '.join(report['stale_artifacts'])}")
    if report["mismatch_artifacts"]:
        print(f"Mismatch: {', '.join(report['mismatch_artifacts'])}")

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
