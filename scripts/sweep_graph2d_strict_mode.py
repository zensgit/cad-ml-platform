#!/usr/bin/env python3
"""Sweep Graph2D training configs and collect strict-mode metrics.

This is a local iteration helper that orchestrates multiple runs of:
  scripts/run_graph2d_pipeline_local.py

Key metric: strict diagnosis accuracy where inference is forced to:
  - strip DXF text/annotation entities
  - mask filename ("masked.dxf")

Artifacts are written under --work-root (default: /tmp/graph2d_strict_sweep_<ts>).
The output directories contain DXF file names and local paths; do not commit them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_eval_overall(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row:
                    continue
                if (row.get("label_cn") or "").strip() != "__overall__":
                    continue
                return {
                    "eval_samples": _safe_int(row.get("total"), 0),
                    "eval_accuracy": _safe_float(row.get("accuracy"), 0.0),
                    "eval_top2_accuracy": _safe_float(row.get("top2_accuracy"), 0.0),
                    "eval_macro_f1": _safe_float(row.get("macro_f1"), 0.0),
                    "eval_weighted_f1": _safe_float(row.get("weighted_f1"), 0.0),
                }
    except Exception:
        return {}
    return {}


def _extract_strict_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    acc = summary.get("accuracy")
    confidence = summary.get("confidence") if isinstance(summary.get("confidence"), dict) else {}
    top_pred = summary.get("top_pred_labels_canon")
    top_pred_label = ""
    top_pred_count = 0
    if isinstance(top_pred, list) and top_pred:
        item = top_pred[0]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            top_pred_label = str(item[0])
            top_pred_count = _safe_int(item[1], 0)

    return {
        "strict_accuracy": _safe_float(acc, -1.0) if acc is not None else -1.0,
        "strict_conf_p50": _safe_float(confidence.get("p50"), 0.0),
        "strict_conf_p90": _safe_float(confidence.get("p90"), 0.0),
        "strict_top_pred_label": top_pred_label,
        "strict_top_pred_count": top_pred_count,
    }


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


def _run(cmd: List[str], env: Dict[str, str], dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"+ {printable}")
    if dry_run:
        return
    merged_env = dict(os.environ)
    merged_env.update({k: str(v) for k, v in env.items() if v is not None})
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, env=merged_env)


@dataclass(frozen=True)
class SweepRun:
    name: str
    args: List[str]
    env: Dict[str, str]


def _default_runs() -> List[SweepRun]:
    """Opinionated run set focused on strict-mode collapse."""
    return [
        SweepRun(
            name="baseline_ce_none_sampler_none",
            args=["--loss", "cross_entropy", "--class-weighting", "none", "--sampler", "none"],
            env={},
        ),
        SweepRun(
            name="baseline_focal_sqrt_balanced",
            args=["--loss", "focal", "--class-weighting", "sqrt", "--sampler", "balanced"],
            env={},
        ),
        SweepRun(
            name="baseline_logit_adj_balanced",
            args=["--loss", "logit_adjusted", "--class-weighting", "none", "--sampler", "balanced"],
            env={},
        ),
        SweepRun(
            name="distill_titleblock_alpha_0_3_focal",
            args=[
                "--distill",
                "--teacher",
                "titleblock",
                "--distill-alpha",
                "0.3",
                "--loss",
                "focal",
                "--class-weighting",
                "sqrt",
                "--sampler",
                "balanced",
            ],
            env={},
        ),
        SweepRun(
            name="distill_titleblock_alpha_0_1_focal",
            args=[
                "--distill",
                "--teacher",
                "titleblock",
                "--distill-alpha",
                "0.1",
                "--loss",
                "focal",
                "--class-weighting",
                "sqrt",
                "--sampler",
                "balanced",
            ],
            env={},
        ),
        SweepRun(
            name="distill_titleblock_alpha_0_3_logit_adj",
            args=[
                "--distill",
                "--teacher",
                "titleblock",
                "--distill-alpha",
                "0.3",
                "--loss",
                "logit_adjusted",
                "--class-weighting",
                "none",
                "--sampler",
                "balanced",
            ],
            env={},
        ),
        SweepRun(
            name="distill_hybrid_alpha_0_3_focal",
            args=[
                "--distill",
                "--teacher",
                "hybrid",
                "--distill-alpha",
                "0.3",
                "--loss",
                "focal",
                "--class-weighting",
                "sqrt",
                "--sampler",
                "balanced",
            ],
            env={},
        ),
        SweepRun(
            name="edge_sage_distill_titleblock_alpha_0_3_focal",
            args=[
                "--model",
                "edge_sage",
                "--distill",
                "--teacher",
                "titleblock",
                "--distill-alpha",
                "0.3",
                "--loss",
                "focal",
                "--class-weighting",
                "sqrt",
                "--sampler",
                "balanced",
            ],
            env={},
        ),
    ]


def _iter_runs(runs: List[SweepRun], max_runs: int) -> Iterable[SweepRun]:
    if max_runs <= 0:
        return runs
    return runs[: max_runs]


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep Graph2D strict-mode configs.")
    parser.add_argument("--dxf-dir", required=True, help="DXF directory to train/eval on.")
    parser.add_argument(
        "--work-root",
        default="",
        help="Work root for sweep artifacts (default: /tmp/graph2d_strict_sweep_<ts>).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch pipeline runs (default: sys.executable).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap for training samples per run (default: 0).",
    )
    parser.add_argument("--diagnose-max-files", type=int, default=200)
    parser.add_argument(
        "--normalize-labels",
        action="store_true",
        help="Normalize fine labels into coarse buckets before cleaning.",
    )
    parser.add_argument(
        "--clean-min-count",
        type=int,
        default=5,
        help="Clean manifest by merging low-frequency labels (default: 5).",
    )
    parser.add_argument(
        "--student-geometry-only",
        action="store_true",
        help="Set DXF_STRIP_TEXT_ENTITIES=true for student graph building.",
    )
    parser.add_argument(
        "--graph-cache-dir",
        default="",
        help="Disk cache directory shared across runs (default: <work_root>/graph_cache).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Limit number of runs from the built-in sweep list (default: 0 = all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running training/eval.",
    )
    args = parser.parse_args()

    dxf_dir = Path(args.dxf_dir)
    if not dxf_dir.exists():
        print(f"DXF dir not found: {dxf_dir}")
        return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    work_root = Path(args.work_root) if args.work_root else Path("/tmp") / f"graph2d_strict_sweep_{stamp}"
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")

    graph_cache_dir = str(args.graph_cache_dir).strip()
    if not graph_cache_dir:
        graph_cache_dir = str(work_root / "graph_cache")

    base_env: Dict[str, str] = {}
    if bool(args.student_geometry_only):
        base_env["DXF_STRIP_TEXT_ENTITIES"] = "true"

    base_pipeline_args = [
        "--dxf-dir",
        str(dxf_dir),
        "--epochs",
        str(int(args.epochs)),
        "--batch-size",
        str(int(args.batch_size)),
        "--hidden-dim",
        str(int(args.hidden_dim)),
        "--lr",
        str(float(args.lr)),
        "--max-samples",
        str(int(args.max_samples)),
        "--diagnose-max-files",
        str(int(args.diagnose_max_files)),
        "--diagnose-no-text-no-filename",
        "--graph-cache",
        "both",
        "--graph-cache-dir",
        str(graph_cache_dir),
        "--student-geometry-only" if bool(args.student_geometry_only) else "",
    ]
    base_pipeline_args = [t for t in base_pipeline_args if t]
    if bool(args.normalize_labels):
        base_pipeline_args.append("--normalize-labels")
    if int(args.clean_min_count) > 0:
        base_pipeline_args.extend(["--clean-min-count", str(int(args.clean_min_count))])

    runs = _default_runs()
    results: List[Dict[str, Any]] = []

    for idx, run in enumerate(_iter_runs(runs, int(args.max_runs)), start=1):
        run_dir = work_root / f"{idx:02d}_{run.name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(args.python),
            "scripts/run_graph2d_pipeline_local.py",
            "--work-dir",
            str(run_dir),
        ]
        cmd.extend(base_pipeline_args)
        cmd.extend(run.args)

        started_at = time.time()
        status = "ok"
        error = ""
        try:
            _run(cmd, env={**base_env, **run.env}, dry_run=bool(args.dry_run))
        except subprocess.CalledProcessError as exc:
            status = "error"
            error = f"pipeline_failed: rc={exc.returncode}"

        elapsed = time.time() - started_at

        pipeline_summary = _read_json(run_dir / "pipeline_summary.json")
        eval_metrics = _read_eval_overall(run_dir / "eval_metrics.csv")
        strict_summary = _read_json(run_dir / "diagnose" / "summary.json")
        strict_metrics = _extract_strict_metrics(strict_summary)

        row: Dict[str, Any] = {
            "run_name": run.name,
            "status": status,
            "error": error,
            "elapsed_seconds": round(elapsed, 3),
            "work_dir": str(run_dir),
            "student_geometry_only": bool(args.student_geometry_only),
            "normalize_labels": bool(args.normalize_labels),
            "clean_min_count": int(args.clean_min_count),
        }

        # Flatten key config tokens for easier sorting.
        row["pipeline_args"] = " ".join(run.args)
        row.update(eval_metrics)
        row.update(strict_metrics)

        manifest = pipeline_summary.get("manifest") if isinstance(pipeline_summary.get("manifest"), dict) else {}
        row["manifest_rows_out"] = _safe_int(manifest.get("rows_out"), 0)
        row["manifest_distinct_labels"] = _safe_int(manifest.get("distinct_labels"), 0)

        results.append(row)

        print(
            json.dumps(
                {
                    "run": run.name,
                    "status": status,
                    "strict_accuracy": row.get("strict_accuracy"),
                    "strict_conf_p50": row.get("strict_conf_p50"),
                    "top_pred": row.get("strict_top_pred_label"),
                },
                ensure_ascii=False,
            )
        )

    results_path = work_root / "sweep_results.csv"
    results_json_path = work_root / "sweep_results.json"
    _write_csv(results_path, results)
    results_json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"results_csv={results_path}")
    print(f"results_json={results_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

