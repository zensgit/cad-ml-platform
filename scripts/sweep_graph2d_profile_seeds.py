#!/usr/bin/env python3
"""Run Graph2D local pipeline for multiple random seeds under one profile.

This script is intended for strict-mode stability checks after a candidate
training profile is selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _extract_top_pred_ratio(summary: Dict[str, Any]) -> Dict[str, Any]:
    sampled_files = _safe_int(summary.get("sampled_files"), 0)
    top_pred = summary.get("top_pred_labels_canon")
    if not isinstance(top_pred, list) or not top_pred:
        top_pred = summary.get("top_pred_labels")
    if not isinstance(top_pred, list) or not top_pred:
        return {"label": "", "count": 0, "ratio": 0.0}
    item = top_pred[0]
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return {"label": "", "count": 0, "ratio": 0.0}
    label = str(item[0])
    count = _safe_int(item[1], 0)
    denom = sampled_files if sampled_files > 0 else 1
    ratio = float(count) / float(denom)
    return {"label": label, "count": count, "ratio": ratio}


def _extract_low_conf_ratio(predictions_csv: Path, threshold: float) -> Dict[str, Any]:
    if not predictions_csv.exists():
        return {"threshold": float(threshold), "count": 0, "total": 0, "ratio": 0.0}
    total = 0
    low = 0
    try:
        with predictions_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                conf = _safe_float(row.get("pred_confidence"), -1.0)
                if conf < 0:
                    continue
                total += 1
                if conf < float(threshold):
                    low += 1
    except Exception:
        return {"threshold": float(threshold), "count": 0, "total": 0, "ratio": 0.0}
    denom = total if total > 0 else 1
    ratio = float(low) / float(denom)
    return {
        "threshold": float(threshold),
        "count": int(low),
        "total": int(total),
        "ratio": float(ratio),
    }


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_yaml_defaults(config_path: str, section: str) -> Dict[str, Any]:
    """Load optional CLI defaults from YAML."""
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception as exc:
        print(f"Warning: yaml unavailable, ignore config {path}: {exc}")
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"Warning: failed to parse config {path}: {exc}")
        return {}
    if not isinstance(payload, dict):
        return {}
    section_payload = payload.get(section)
    data = section_payload if isinstance(section_payload, dict) else payload
    if not isinstance(data, dict):
        return {}
    return {str(k).replace("-", "_"): v for k, v in data.items()}


def _apply_config_defaults(
    parser: argparse.ArgumentParser, config_path: str, section: str
) -> None:
    defaults = _load_yaml_defaults(config_path, section)
    if not defaults:
        return
    valid_keys = {action.dest for action in parser._actions}
    filtered = {k: v for k, v in defaults.items() if k in valid_keys}
    unknown = sorted(set(defaults.keys()) - set(filtered.keys()))
    if unknown:
        print(
            f"Warning: ignored unknown keys in {config_path} ({section}): "
            + ", ".join(unknown)
        )
    if filtered:
        parser.set_defaults(**filtered)


def _parse_seeds(raw: str) -> List[int]:
    tokens = [t.strip() for t in str(raw or "").split(",") if t.strip()]
    out: List[int] = []
    for token in tokens:
        out.append(int(token))
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run(cmd: List[str], dry_run: bool = False) -> None:
    printable = " ".join(cmd)
    print(f"+ {printable}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _run_with_retries(
    cmd: List[str],
    *,
    dry_run: bool,
    retry_failures: int,
    retry_backoff_seconds: float,
) -> Dict[str, Any]:
    max_attempts = 1 + max(0, int(retry_failures))
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        try:
            _run(cmd, dry_run=dry_run)
            return {
                "status": "ok",
                "error": "",
                "attempts": attempts,
                "return_code": 0,
            }
        except subprocess.CalledProcessError as exc:
            if attempts >= max_attempts:
                return {
                    "status": "error",
                    "error": f"pipeline_failed: rc={exc.returncode}",
                    "attempts": attempts,
                    "return_code": int(exc.returncode),
                }
            print(
                "Warning: pipeline run failed"
                f" (attempt {attempts}/{max_attempts}, rc={exc.returncode}); retrying."
            )
            delay = max(0.0, float(retry_backoff_seconds))
            if delay > 0 and not dry_run:
                time.sleep(delay)
    return {
        "status": "error",
        "error": "pipeline_failed: rc=unknown",
        "attempts": attempts,
        "return_code": -1,
    }


def _evaluate_gate(
    *,
    rows: List[Dict[str, Any]],
    strict_accuracy_mean: float,
    strict_accuracy_min: float,
    min_strict_accuracy_mean: float,
    min_strict_accuracy_min: float,
    min_manifest_distinct_labels: int,
    max_strict_top_pred_ratio: float,
    max_strict_low_conf_ratio: float,
    require_all_ok: bool,
) -> Dict[str, Any]:
    failures: List[str] = []
    num_errors = sum(1 for row in rows if str(row.get("status", "")) != "ok")
    if bool(require_all_ok) and num_errors > 0:
        failures.append(f"require_all_ok: error_runs={num_errors}")
    if float(min_strict_accuracy_mean) >= 0 and float(strict_accuracy_mean) < float(
        min_strict_accuracy_mean
    ):
        failures.append(
            "strict_accuracy_mean"
            f": {strict_accuracy_mean:.6f} < {float(min_strict_accuracy_mean):.6f}"
        )
    if float(min_strict_accuracy_min) >= 0 and float(strict_accuracy_min) < float(
        min_strict_accuracy_min
    ):
        failures.append(
            "strict_accuracy_min"
            f": {strict_accuracy_min:.6f} < {float(min_strict_accuracy_min):.6f}"
        )
    if int(min_manifest_distinct_labels) > 0:
        bad_rows = [
            row
            for row in rows
            if _safe_int(row.get("manifest_distinct_labels"), 0)
            < int(min_manifest_distinct_labels)
        ]
        if bad_rows:
            bad_desc = ",".join(
                f"{_safe_int(row.get('seed'), -1)}:{_safe_int(row.get('manifest_distinct_labels'), 0)}"
                for row in bad_rows
            )
            failures.append(
                "manifest_distinct_labels: "
                f"below {int(min_manifest_distinct_labels)} for seeds [{bad_desc}]"
            )
    if float(max_strict_top_pred_ratio) >= 0:
        bad_rows = [
            row
            for row in rows
            if _safe_float(row.get("strict_top_pred_ratio"), 0.0)
            > float(max_strict_top_pred_ratio)
        ]
        if bad_rows:
            bad_desc = ",".join(
                f"{_safe_int(row.get('seed'), -1)}:{_safe_float(row.get('strict_top_pred_ratio'), 0.0):.3f}"
                for row in bad_rows
            )
            failures.append(
                "strict_top_pred_ratio: "
                f"above {float(max_strict_top_pred_ratio):.3f} for seeds [{bad_desc}]"
            )
    if float(max_strict_low_conf_ratio) >= 0:
        bad_rows = [
            row
            for row in rows
            if _safe_float(row.get("strict_low_conf_ratio"), 0.0)
            > float(max_strict_low_conf_ratio)
        ]
        if bad_rows:
            bad_desc = ",".join(
                f"{_safe_int(row.get('seed'), -1)}:{_safe_float(row.get('strict_low_conf_ratio'), 0.0):.3f}"
                for row in bad_rows
            )
            failures.append(
                "strict_low_conf_ratio: "
                f"above {float(max_strict_low_conf_ratio):.3f} for seeds [{bad_desc}]"
            )

    gate_enabled = bool(require_all_ok) or float(min_strict_accuracy_mean) >= 0 or float(
        min_strict_accuracy_min
    ) >= 0 or int(min_manifest_distinct_labels) > 0 or float(
        max_strict_top_pred_ratio
    ) >= 0 or float(max_strict_low_conf_ratio) >= 0
    return {
        "enabled": gate_enabled,
        "passed": len(failures) == 0,
        "failures": failures,
        "require_all_ok": bool(require_all_ok),
        "min_strict_accuracy_mean": float(min_strict_accuracy_mean),
        "min_strict_accuracy_min": float(min_strict_accuracy_min),
        "min_manifest_distinct_labels": int(min_manifest_distinct_labels),
        "max_strict_top_pred_ratio": float(max_strict_top_pred_ratio),
        "max_strict_low_conf_ratio": float(max_strict_low_conf_ratio),
        "num_runs": len(rows),
        "num_error_runs": int(num_errors),
    }


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        default="config/graph2d_seed_gate.yaml",
        help="YAML config path for seed sweep defaults.",
    )
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Sweep Graph2D local pipeline over random seeds for one profile.",
        parents=[pre_parser],
    )
    parser.add_argument("--dxf-dir", default="", help="DXF directory")
    parser.add_argument(
        "--training-profile",
        default="strict_node23_edgesage_v1",
        help="Profile passed to run_graph2d_pipeline_local.py",
    )
    parser.add_argument(
        "--manifest-label-mode",
        choices=["filename", "parent_dir"],
        default="filename",
        help="Weak-label mode passed to run_graph2d_pipeline_local.py (default: filename).",
    )
    parser.add_argument(
        "--seeds",
        default="7,21,42",
        help="Comma-separated random seeds (default: 7,21,42)",
    )
    parser.add_argument(
        "--work-root",
        default="",
        help="Output root (default: /tmp/graph2d_profile_seed_sweep_<ts>)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter (default: sys.executable)",
    )
    parser.add_argument(
        "--diagnose-max-files",
        type=int,
        default=200,
        help="Diagnosis max files (default: 200)",
    )
    parser.add_argument(
        "--min-label-confidence",
        type=float,
        default=0.8,
        help="Manifest label-confidence filter passed to pipeline (default: 0.8).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional max training samples per seed run (default: 0 = no cap).",
    )
    parser.add_argument(
        "--force-normalize-labels",
        choices=["auto", "true", "false"],
        default="auto",
        help="Post-profile normalize-labels override passed to pipeline.",
    )
    parser.add_argument(
        "--force-clean-min-count",
        type=int,
        default=-1,
        help="Post-profile clean_min_count override passed to pipeline.",
    )
    parser.add_argument(
        "--retry-failures",
        type=int,
        default=0,
        help="Retry failed seed runs up to N times (default: 0).",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=0.0,
        help="Sleep before retrying a failed seed run (default: 0).",
    )
    parser.add_argument(
        "--min-strict-accuracy-mean",
        type=float,
        default=-1.0,
        help="Fail gate when strict mean accuracy is below this value (default: disabled).",
    )
    parser.add_argument(
        "--min-strict-accuracy-min",
        type=float,
        default=-1.0,
        help="Fail gate when strict min accuracy is below this value (default: disabled).",
    )
    parser.add_argument(
        "--min-manifest-distinct-labels",
        type=int,
        default=-1,
        help=(
            "Fail gate when any seed run has distinct labels below this threshold "
            "(default: disabled)."
        ),
    )
    parser.add_argument(
        "--max-strict-top-pred-ratio",
        type=float,
        default=-1.0,
        help=(
            "Fail gate when any run has top-prediction ratio above this threshold "
            "(default: disabled)."
        ),
    )
    parser.add_argument(
        "--strict-low-confidence-threshold",
        type=float,
        default=0.2,
        help=(
            "Confidence threshold used to calculate low-confidence ratio "
            "(default: 0.2)."
        ),
    )
    parser.add_argument(
        "--max-strict-low-conf-ratio",
        type=float,
        default=-1.0,
        help=(
            "Fail gate when any run has low-confidence ratio above this threshold "
            "(default: disabled)."
        ),
    )
    parser.add_argument(
        "--require-all-ok",
        action="store_true",
        help="Fail gate when any seed run exits with error.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    _apply_config_defaults(parser, str(pre_args.config), "graph2d_seed_sweep")
    args = parser.parse_args()

    if not str(args.dxf_dir).strip():
        print("DXF dir is required. Pass --dxf-dir or provide dxf_dir in config.")
        return 2
    dxf_dir = Path(args.dxf_dir)
    if not dxf_dir.exists():
        print(f"DXF dir not found: {dxf_dir}")
        return 2

    seeds = _parse_seeds(args.seeds)
    if not seeds:
        print("No seeds provided.")
        return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    work_root = (
        Path(args.work_root)
        if args.work_root
        else Path("/tmp") / f"graph2d_profile_seed_sweep_{stamp}"
    )
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        run_dir = work_root / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        cmd = [
            str(args.python),
            "scripts/run_graph2d_pipeline_local.py",
            "--dxf-dir",
            str(dxf_dir),
            "--work-dir",
            str(run_dir),
            "--training-profile",
            str(args.training_profile),
            "--manifest-label-mode",
            str(args.manifest_label_mode),
            "--seed",
            str(int(seed)),
            "--diagnose-max-files",
            str(int(args.diagnose_max_files)),
            "--min-label-confidence",
            str(float(args.min_label_confidence)),
        ]
        if str(args.force_normalize_labels) != "auto":
            cmd.extend(["--force-normalize-labels", str(args.force_normalize_labels)])
        if int(args.force_clean_min_count) >= 0:
            cmd.extend(["--force-clean-min-count", str(int(args.force_clean_min_count))])
        if int(args.max_samples) > 0:
            cmd.extend(["--max-samples", str(int(args.max_samples))])

        run_result = _run_with_retries(
            cmd=cmd,
            dry_run=bool(args.dry_run),
            retry_failures=int(args.retry_failures),
            retry_backoff_seconds=float(args.retry_backoff_seconds),
        )
        status = str(run_result.get("status", "error"))
        error = str(run_result.get("error", ""))
        attempts = int(run_result.get("attempts", 0))
        return_code = int(run_result.get("return_code", -1))

        elapsed = time.time() - started
        pipeline_summary = _read_json(run_dir / "pipeline_summary.json")
        strict_summary = _read_json(run_dir / "diagnose" / "summary.json")
        top_pred = _extract_top_pred_ratio(strict_summary)
        low_conf = _extract_low_conf_ratio(
            run_dir / "diagnose" / "predictions.csv",
            threshold=float(args.strict_low_confidence_threshold),
        )

        row: Dict[str, Any] = {
            "seed": int(seed),
            "status": status,
            "error": error,
            "attempts": attempts,
            "return_code": return_code,
            "elapsed_seconds": round(elapsed, 3),
            "work_dir": str(run_dir),
            "training_profile": str(args.training_profile),
            "strict_accuracy": _safe_float(strict_summary.get("accuracy"), -1.0),
            "strict_conf_p50": (
                _safe_float((strict_summary.get("confidence") or {}).get("p50"), 0.0)
                if isinstance(strict_summary.get("confidence"), dict)
                else 0.0
            ),
            "strict_conf_p90": (
                _safe_float((strict_summary.get("confidence") or {}).get("p90"), 0.0)
                if isinstance(strict_summary.get("confidence"), dict)
                else 0.0
            ),
            "strict_top_pred_label": str(top_pred.get("label") or ""),
            "strict_top_pred_count": _safe_int(top_pred.get("count"), 0),
            "strict_top_pred_ratio": _safe_float(top_pred.get("ratio"), 0.0),
            "strict_low_conf_threshold": _safe_float(low_conf.get("threshold"), 0.0),
            "strict_low_conf_count": _safe_int(low_conf.get("count"), 0),
            "strict_low_conf_total": _safe_int(low_conf.get("total"), 0),
            "strict_low_conf_ratio": _safe_float(low_conf.get("ratio"), 0.0),
            "training_model": str(
                (pipeline_summary.get("training") or {}).get("model", "")
            ),
            "manifest_distinct_labels": (
                _safe_int((pipeline_summary.get("manifest") or {}).get("distinct_labels"), 0)
                if isinstance(pipeline_summary.get("manifest"), dict)
                else 0
            ),
            "training_node_dim": (
                int((pipeline_summary.get("training") or {}).get("node_dim", 0))
                if isinstance(pipeline_summary.get("training"), dict)
                else 0
            ),
            "training_hidden_dim": (
                int((pipeline_summary.get("training") or {}).get("hidden_dim", 0))
                if isinstance(pipeline_summary.get("training"), dict)
                else 0
            ),
            "training_epochs": (
                int((pipeline_summary.get("training") or {}).get("epochs", 0))
                if isinstance(pipeline_summary.get("training"), dict)
                else 0
            ),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    acc_values = [r["strict_accuracy"] for r in rows if r["strict_accuracy"] >= 0]
    strict_accuracy_mean = round(sum(acc_values) / len(acc_values), 6) if acc_values else -1.0
    strict_accuracy_min = round(min(acc_values), 6) if acc_values else -1.0
    strict_accuracy_max = round(max(acc_values), 6) if acc_values else -1.0
    top_pred_ratios = [
        _safe_float(r.get("strict_top_pred_ratio"), 0.0)
        for r in rows
        if _safe_float(r.get("strict_top_pred_ratio"), 0.0) > 0
    ]
    low_conf_ratios = [
        _safe_float(r.get("strict_low_conf_ratio"), 0.0)
        for r in rows
        if _safe_int(r.get("strict_low_conf_total"), 0) > 0
    ]
    manifest_distinct_values = [
        _safe_int(r.get("manifest_distinct_labels"), 0)
        for r in rows
        if _safe_int(r.get("manifest_distinct_labels"), 0) > 0
    ]

    summary: Dict[str, Any] = {
        "status": "ok" if rows else "empty",
        "config": str(args.config),
        "training_profile": str(args.training_profile),
        "manifest_label_mode": str(args.manifest_label_mode),
        "seeds": seeds,
        "num_runs": len(rows),
        "num_success_runs": sum(1 for r in rows if str(r.get("status", "")) == "ok"),
        "num_error_runs": sum(1 for r in rows if str(r.get("status", "")) != "ok"),
        "num_retried_runs": sum(1 for r in rows if int(r.get("attempts", 0)) > 1),
        "retry_failures": int(max(0, int(args.retry_failures))),
        "retry_backoff_seconds": float(max(0.0, float(args.retry_backoff_seconds))),
        "max_samples": int(max(0, int(args.max_samples))),
        "min_label_confidence": float(args.min_label_confidence),
        "force_normalize_labels": str(args.force_normalize_labels),
        "force_clean_min_count": int(args.force_clean_min_count),
        "strict_accuracy_mean": strict_accuracy_mean,
        "strict_accuracy_min": strict_accuracy_min,
        "strict_accuracy_max": strict_accuracy_max,
        "strict_top_pred_ratio_mean": (
            round(sum(top_pred_ratios) / len(top_pred_ratios), 6)
            if top_pred_ratios
            else 0.0
        ),
        "strict_top_pred_ratio_max": (
            round(max(top_pred_ratios), 6) if top_pred_ratios else 0.0
        ),
        "strict_low_conf_threshold": float(args.strict_low_confidence_threshold),
        "strict_low_conf_ratio_mean": (
            round(sum(low_conf_ratios) / len(low_conf_ratios), 6)
            if low_conf_ratios
            else 0.0
        ),
        "strict_low_conf_ratio_max": (
            round(max(low_conf_ratios), 6) if low_conf_ratios else 0.0
        ),
        "manifest_distinct_labels_min": (
            min(manifest_distinct_values) if manifest_distinct_values else 0
        ),
        "manifest_distinct_labels_max": (
            max(manifest_distinct_values) if manifest_distinct_values else 0
        ),
        "work_root": str(work_root),
    }

    gate = _evaluate_gate(
        rows=rows,
        strict_accuracy_mean=float(strict_accuracy_mean),
        strict_accuracy_min=float(strict_accuracy_min),
        min_strict_accuracy_mean=float(args.min_strict_accuracy_mean),
        min_strict_accuracy_min=float(args.min_strict_accuracy_min),
        min_manifest_distinct_labels=int(args.min_manifest_distinct_labels),
        max_strict_top_pred_ratio=float(args.max_strict_top_pred_ratio),
        max_strict_low_conf_ratio=float(args.max_strict_low_conf_ratio),
        require_all_ok=bool(args.require_all_ok),
    )
    summary["gate"] = gate
    if bool(gate.get("enabled")) and not bool(gate.get("passed")):
        summary["status"] = "gate_failed"

    results_csv = work_root / "seed_sweep_results.csv"
    results_json = work_root / "seed_sweep_results.json"
    summary_json = work_root / "seed_sweep_summary.json"

    _write_csv(results_csv, rows)
    results_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"results_csv={results_csv}")
    print(f"results_json={results_json}")
    print(f"summary_json={summary_json}")
    print(f"gate={json.dumps(gate, ensure_ascii=False)}")
    if bool(gate.get("enabled")) and not bool(gate.get("passed")):
        print("Seed sweep gate failed.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
