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


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _evaluate_gate(
    *,
    rows: List[Dict[str, Any]],
    strict_accuracy_mean: float,
    strict_accuracy_min: float,
    min_strict_accuracy_mean: float,
    min_strict_accuracy_min: float,
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

    gate_enabled = bool(require_all_ok) or float(min_strict_accuracy_mean) >= 0 or float(
        min_strict_accuracy_min
    ) >= 0
    return {
        "enabled": gate_enabled,
        "passed": len(failures) == 0,
        "failures": failures,
        "require_all_ok": bool(require_all_ok),
        "min_strict_accuracy_mean": float(min_strict_accuracy_mean),
        "min_strict_accuracy_min": float(min_strict_accuracy_min),
        "num_runs": len(rows),
        "num_error_runs": int(num_errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep Graph2D local pipeline over random seeds for one profile."
    )
    parser.add_argument("--dxf-dir", required=True, help="DXF directory")
    parser.add_argument(
        "--training-profile",
        default="strict_node23_edgesage_v1",
        help="Profile passed to run_graph2d_pipeline_local.py",
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
        "--require-all-ok",
        action="store_true",
        help="Fail gate when any seed run exits with error.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

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
        status = "ok"
        error = ""
        try:
            cmd = [
                str(args.python),
                "scripts/run_graph2d_pipeline_local.py",
                "--dxf-dir",
                str(dxf_dir),
                "--work-dir",
                str(run_dir),
                "--training-profile",
                str(args.training_profile),
                "--seed",
                str(int(seed)),
                "--diagnose-max-files",
                str(int(args.diagnose_max_files)),
            ]
            _run(cmd, dry_run=bool(args.dry_run))
        except subprocess.CalledProcessError as exc:
            status = "error"
            error = f"pipeline_failed: rc={exc.returncode}"

        elapsed = time.time() - started
        pipeline_summary = _read_json(run_dir / "pipeline_summary.json")
        strict_summary = _read_json(run_dir / "diagnose" / "summary.json")

        row: Dict[str, Any] = {
            "seed": int(seed),
            "status": status,
            "error": error,
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
            "training_model": str(
                (pipeline_summary.get("training") or {}).get("model", "")
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

    summary: Dict[str, Any] = {
        "status": "ok" if rows else "empty",
        "training_profile": str(args.training_profile),
        "seeds": seeds,
        "num_runs": len(rows),
        "num_success_runs": sum(1 for r in rows if str(r.get("status", "")) == "ok"),
        "num_error_runs": sum(1 for r in rows if str(r.get("status", "")) != "ok"),
        "strict_accuracy_mean": strict_accuracy_mean,
        "strict_accuracy_min": strict_accuracy_min,
        "strict_accuracy_max": strict_accuracy_max,
        "work_root": str(work_root),
    }

    gate = _evaluate_gate(
        rows=rows,
        strict_accuracy_mean=float(strict_accuracy_mean),
        strict_accuracy_min=float(strict_accuracy_min),
        min_strict_accuracy_mean=float(args.min_strict_accuracy_mean),
        min_strict_accuracy_min=float(args.min_strict_accuracy_min),
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
