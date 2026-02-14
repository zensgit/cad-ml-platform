#!/usr/bin/env python3
"""Run a local Graph2D pipeline on a DXF directory.

This script is a thin orchestrator around existing building blocks:
1) Build a weak-labeled manifest from DXF filenames
2) Filter the manifest by weak-label confidence
3) Train a Graph2D checkpoint
4) Evaluate the checkpoint on a validation split
5) Diagnose the checkpoint on a sampled subset (optionally scoring against weak labels)

Outputs are written under --work-dir (default: /tmp/graph2d_pipeline_local_<timestamp>).
The produced artifacts are intended for local iteration and should generally not be
committed (they include file names and local paths).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

TRAINING_PROFILES: Dict[str, Dict[str, Any]] = {
    "strict_node23_edgesage_v1": {
        "model": "edge_sage",
        "node_dim": 23,
        "hidden_dim": 128,
        "epochs": 10,
        "loss": "focal",
        "class_weighting": "sqrt",
        "sampler": "balanced",
        "distill": True,
        "teacher": "titleblock",
        "distill_alpha": 0.1,
        "distill_temp": 3.0,
        "distill_mask_filename": "auto",
        "student_geometry_only": True,
        "diagnose_no_text_no_filename": True,
        "normalize_labels": True,
        "clean_min_count": 5,
        "dxf_enhanced_keypoints": "true",
        "dxf_edge_augment_knn_k": 0,
        "dxf_edge_augment_strategy": "union_all",
        "dxf_eps_scale": 0.001,
    }
}


def _run(cmd: List[str]) -> None:
    printable = " ".join(cmd)
    print(f"+ {printable}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row]
        fieldnames = list(reader.fieldnames or [])
    return fieldnames, rows


def _write_csv_rows(
    path: Path, fieldnames: List[str], rows: List[Dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _filter_manifest_by_confidence(
    input_csv: Path, output_csv: Path, *, min_confidence: float
) -> Dict[str, Any]:
    fieldnames, rows = _load_csv_rows(input_csv)
    if not rows:
        raise RuntimeError(f"empty_manifest: {input_csv}")

    kept: List[Dict[str, str]] = []
    dropped = 0

    for row in rows:
        label = (row.get("label_cn") or "").strip()
        conf = _safe_float(row.get("label_confidence"), 0.0)
        status = (row.get("label_status") or "").strip()
        if not label:
            dropped += 1
            continue
        if conf < float(min_confidence):
            dropped += 1
            continue
        # Prefer keeping only manifest rows that produced a stable label.
        if status and status not in {"matched", "dir_label"}:
            # Keep "matched" labels; allow parent-dir labels for synthetic sets.
            dropped += 1
            continue
        kept.append(row)

    if not kept:
        raise RuntimeError(
            f"no_rows_after_filter: min_conf={min_confidence} input={input_csv}"
        )

    _write_csv_rows(output_csv, fieldnames, kept)
    distinct_labels = sorted(
        {(row.get("label_cn") or "").strip() for row in kept if row}
    )
    return {
        "rows_in": len(rows),
        "rows_out": len(kept),
        "dropped": int(dropped),
        "distinct_labels": len(distinct_labels),
        "min_confidence": float(min_confidence),
    }


def _resolve_distill_mask(teacher: str, mode: str) -> bool:
    """Resolve whether to pass --distill-mask-filename to train_2d_graph.py.

    mode:
    - auto: mask for hybrid/titleblock teachers (prevents filename leakage)
    - true: always mask
    - false: never mask
    """

    token = str(mode or "").strip().lower()
    if token in {"true", "1", "yes", "on"}:
        return True
    if token in {"false", "0", "no", "off"}:
        return False
    return str(teacher or "").strip().lower() in {"hybrid", "titleblock"}


def _build_train_cmd(
    *,
    python: str,
    manifest_csv: Path,
    dxf_dir: Path,
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> List[str]:
    """Build the train_2d_graph.py command based on pipeline args."""

    train_cmd = [
        python,
        "scripts/train_2d_graph.py",
        "--manifest",
        str(manifest_csv),
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
        "--node-dim",
        str(int(getattr(args, "node_dim", 19))),
        "--model",
        str(args.model),
        "--loss",
        str(args.loss),
        "--class-weighting",
        str(args.class_weighting),
        "--sampler",
        str(args.sampler),
        "--seed",
        str(int(args.seed)),
        "--output",
        str(checkpoint_path),
        "--device",
        str(args.device),
        "--dxf-max-nodes",
        str(int(args.dxf_max_nodes)),
        "--dxf-sampling-strategy",
        str(args.dxf_sampling_strategy),
        "--dxf-sampling-seed",
        str(int(args.dxf_sampling_seed)),
        "--dxf-text-priority-ratio",
        str(float(args.dxf_text_priority_ratio)),
    ]
    if getattr(args, "dxf_frame_priority_ratio", None) is not None:
        train_cmd.extend(
            ["--dxf-frame-priority-ratio", str(float(args.dxf_frame_priority_ratio))]
        )
    if getattr(args, "dxf_long_line_ratio", None) is not None:
        train_cmd.extend(
            ["--dxf-long-line-ratio", str(float(args.dxf_long_line_ratio))]
        )
    if int(args.max_samples) > 0:
        train_cmd.extend(["--max-samples", str(int(args.max_samples))])

    if bool(getattr(args, "distill", False)):
        train_cmd.append("--distill")
        train_cmd.extend(["--teacher", str(args.teacher)])
        train_cmd.extend(["--distill-alpha", str(float(args.distill_alpha))])
        train_cmd.extend(["--distill-temp", str(float(args.distill_temp))])
        if _resolve_distill_mask(str(args.teacher), str(args.distill_mask_filename)):
            train_cmd.append("--distill-mask-filename")

    return train_cmd


def _build_diagnose_cmd(
    *,
    python: str,
    dxf_dir: Path,
    checkpoint_path: Path,
    manifest_csv: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        python,
        "scripts/diagnose_graph2d_on_dxf_dir.py",
        "--dxf-dir",
        str(dxf_dir),
        "--model-path",
        str(checkpoint_path),
        "--manifest-csv",
        str(manifest_csv),
        "--true-label-min-confidence",
        str(float(args.min_label_confidence)),
        "--max-files",
        str(int(args.diagnose_max_files)),
        "--seed",
        str(int(args.seed)),
        "--output-dir",
        str(out_dir),
    ]

    # Optional strict mode for stable regression metrics.
    if bool(getattr(args, "diagnose_no_text_no_filename", False)):
        cmd.extend(["--strip-text-entities", "--mask-filename"])

    return cmd


def _apply_training_profile(args: argparse.Namespace) -> argparse.Namespace:
    token = str(getattr(args, "training_profile", "none") or "none").strip().lower()
    if token in {"", "none"}:
        args.training_profile = "none"
        return args

    profile = TRAINING_PROFILES.get(token)
    if not profile:
        raise ValueError(f"unknown training profile: {token}")

    for key, value in profile.items():
        setattr(args, key, value)
    args.training_profile = token
    return args


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Graph2D pipeline (manifest -> train -> eval -> diagnose)."
    )
    parser.add_argument(
        "--dxf-dir", required=True, help="DXF directory to scan/train on."
    )
    parser.add_argument(
        "--training-profile",
        choices=["none", *sorted(TRAINING_PROFILES.keys())],
        default="none",
        help=(
            "Apply an opinionated training profile. "
            "When not 'none', profile values override related CLI args."
        ),
    )
    parser.add_argument(
        "--work-dir",
        default="",
        help="Working directory for artifacts (default: /tmp/graph2d_pipeline_local_<ts>).",
    )
    parser.add_argument(
        "--min-label-confidence",
        type=float,
        default=0.8,
        help="Minimum weak-label confidence to keep manifest rows (default: 0.8).",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--node-dim",
        type=int,
        default=19,
        help=(
            "Node feature dimension passed to train_2d_graph.py (default: 19). "
            "Use >19 to enable appended extra node features in dataset_2d."
        ),
    )
    parser.add_argument(
        "--model",
        choices=["gcn", "edge_sage"],
        default="gcn",
        help="Graph2D model family to train (default: gcn).",
    )
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "focal", "logit_adjusted"],
        default="focal",
        help="Loss function (default: focal).",
    )
    parser.add_argument(
        "--class-weighting",
        choices=["none", "inverse", "sqrt"],
        default="sqrt",
        help="Optional class weighting strategy (default: sqrt).",
    )
    parser.add_argument(
        "--sampler",
        choices=["none", "balanced"],
        default="balanced",
        help="Optional training sampler (default: balanced).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap for training samples (default: 0 = no cap).",
    )
    parser.add_argument(
        "--normalize-labels",
        action="store_true",
        help=(
            "Normalize fine-grained labels into coarse buckets via "
            "scripts/normalize_dxf_label_manifest.py."
        ),
    )
    parser.add_argument(
        "--normalize-default-label",
        default="other",
        help="Fallback label used by label normalization (default: other).",
    )
    parser.add_argument(
        "--clean-min-count",
        type=int,
        default=0,
        help=(
            "If >0, apply scripts/clean_dxf_label_manifest.py with this min-count "
            "to merge/drop low-frequency labels."
        ),
    )
    parser.add_argument(
        "--clean-drop-low",
        action="store_true",
        help="When cleaning labels, drop low-frequency classes instead of mapping to other.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for training (default: cpu).",
    )
    parser.add_argument(
        "--dxf-max-nodes",
        type=int,
        default=200,
        help="DXF_MAX_NODES override for importance sampling (default: 200).",
    )
    parser.add_argument(
        "--dxf-sampling-strategy",
        choices=["importance", "random", "hybrid"],
        default="importance",
        help="DXF_SAMPLING_STRATEGY override (default: importance).",
    )
    parser.add_argument(
        "--dxf-sampling-seed",
        type=int,
        default=42,
        help="DXF_SAMPLING_SEED override (default: 42).",
    )
    parser.add_argument(
        "--dxf-text-priority-ratio",
        type=float,
        default=0.3,
        help="DXF_TEXT_PRIORITY_RATIO override (default: 0.3).",
    )
    parser.add_argument(
        "--dxf-frame-priority-ratio",
        type=float,
        default=None,
        help=(
            "DXF_FRAME_PRIORITY_RATIO override (caps border/titleblock frame entities). "
            "Default: unset (no cap). When --student-geometry-only is set, defaults "
            "to 0.1."
        ),
    )
    parser.add_argument(
        "--dxf-long-line-ratio",
        type=float,
        default=None,
        help=(
            "DXF_LONG_LINE_RATIO override (caps non-frame long lines). "
            "Default: unset (no cap). When --student-geometry-only is set, defaults "
            "to 0.4."
        ),
    )
    parser.add_argument(
        "--dxf-eps-scale",
        type=float,
        default=0.001,
        help=(
            "DXF_EPS_SCALE override for epsilon-adjacency distance "
            "(eps = max_dim * scale). Default: 0.001."
        ),
    )
    parser.add_argument(
        "--dxf-edge-augment-knn-k",
        type=int,
        default=None,
        help=(
            "DXF_EDGE_AUGMENT_KNN_K override (adds kNN edges even when epsilon-adjacency "
            "edges exist). Default: auto (0 normally; geometry-only defaults to 0 when "
            "enhanced keypoints are enabled, else 8)."
        ),
    )
    parser.add_argument(
        "--dxf-edge-augment-strategy",
        choices=["auto", "union_all", "isolates_only"],
        default="auto",
        help=(
            "Controls DXF_EDGE_AUGMENT_STRATEGY for kNN augmentation. "
            "union_all adds kNN edges for all nodes; isolates_only only adds edges for "
            "isolated nodes (degree=0) in the epsilon-adjacency graph. "
            "Default: auto (union_all)."
        ),
    )
    parser.add_argument(
        "--dxf-enhanced-keypoints",
        choices=["auto", "true", "false"],
        default="auto",
        help=(
            "Controls DXF_ENHANCED_KEYPOINTS (adds circle/arc keypoints to improve "
            "epsilon-adjacency). Default: auto (enabled for --student-geometry-only)."
        ),
    )
    parser.add_argument(
        "--diagnose-max-files",
        type=int,
        default=200,
        help="Max files to sample for diagnosis (default: 200).",
    )
    parser.add_argument(
        "--diagnose-no-text-no-filename",
        action="store_true",
        help=(
            "Run diagnosis in strict mode (strip DXF text entities + mask filename) "
            "to simulate production conditions."
        ),
    )
    parser.add_argument(
        "--graph-cache",
        choices=["none", "memory", "disk", "both"],
        default="memory",
        help="Enable DXFManifestDataset graph caching during training (default: memory).",
    )
    parser.add_argument(
        "--graph-cache-dir",
        default="",
        help="Optional directory for disk graph cache (used when --graph-cache=disk|both).",
    )
    parser.add_argument(
        "--graph-cache-max-items",
        type=int,
        default=0,
        help="Max cached graphs in memory (0 = unlimited).",
    )
    parser.add_argument(
        "--student-geometry-only",
        action="store_true",
        help=(
            "Set DXF_STRIP_TEXT_ENTITIES=true for graph building during training/eval "
            "(simulate geometry-only student input)."
        ),
    )
    parser.add_argument(
        "--empty-edge-fallback",
        choices=["fully_connected", "knn"],
        default=None,
        help=(
            "Fallback edge strategy when a DXF graph has no edges. Default: auto "
            "(fully_connected normally; knn when --student-geometry-only is set)."
        ),
    )
    parser.add_argument(
        "--empty-edge-knn-k",
        type=int,
        default=8,
        help="k for kNN fallback when --empty-edge-fallback=knn (default: 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--distill",
        action="store_true",
        help="Enable knowledge distillation training (student=Graph2D, teacher=rules/text).",
    )
    parser.add_argument(
        "--teacher",
        choices=["filename", "titleblock", "hybrid"],
        default="hybrid",
        help="Teacher model type for distillation (default: hybrid).",
    )
    parser.add_argument(
        "--distill-alpha",
        type=float,
        default=0.3,
        help="Distillation alpha (CE weight) (default: 0.3).",
    )
    parser.add_argument(
        "--distill-temp",
        type=float,
        default=3.0,
        help="Distillation temperature (default: 3.0).",
    )
    parser.add_argument(
        "--distill-mask-filename",
        choices=["auto", "true", "false"],
        default="auto",
        help=(
            "Whether to mask the filename when calling hybrid/titleblock teachers "
            "(prevents filename label leakage). Default: auto."
        ),
    )
    args = parser.parse_args()
    args = _apply_training_profile(args)

    dxf_dir = Path(args.dxf_dir)
    if not dxf_dir.exists():
        print(f"DXF dir not found: {dxf_dir}")
        return 2

    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        work_dir = Path("/tmp") / f"graph2d_pipeline_local_{stamp}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Keep artifacts out of git by default; the repo-level .gitignore might not
    # cover nested work dirs in reports/experiments, so we create a local ignore.
    (work_dir / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")

    manifest_csv = work_dir / "manifest.csv"
    manifest_filtered_csv = work_dir / "manifest.filtered.csv"
    manifest_normalized_csv = work_dir / "manifest.normalized.csv"
    manifest_cleaned_csv = work_dir / "manifest.cleaned.csv"
    checkpoint_path = work_dir / "graph2d_trained.pth"
    eval_metrics_csv = work_dir / "eval_metrics.csv"
    eval_errors_csv = work_dir / "eval_errors.csv"
    diagnose_dir = work_dir / "diagnose"
    diagnose_summary_json = diagnose_dir / "summary.json"

    python = sys.executable
    print(f"work_dir={work_dir}")

    # Dataset graph caching (train/eval). Memory cache is per-process; disk cache
    # allows sharing graph build results across subprocess steps.
    graph_cache = str(args.graph_cache).strip().lower()
    if graph_cache == "none":
        os.environ["DXF_MANIFEST_DATASET_CACHE"] = "none"
    else:
        os.environ["DXF_MANIFEST_DATASET_CACHE"] = graph_cache
        os.environ["DXF_MANIFEST_DATASET_CACHE_MAX_ITEMS"] = str(
            int(args.graph_cache_max_items)
        )
        if graph_cache in {"disk", "both"}:
            cache_dir = str(args.graph_cache_dir or "").strip()
            if not cache_dir:
                cache_dir = str(work_dir / "graph_cache")
            os.environ["DXF_MANIFEST_DATASET_CACHE_DIR"] = cache_dir

    # Empty-edge fallback for DXF graphs.
    if getattr(args, "empty_edge_fallback", None) is None:
        args.empty_edge_fallback = (
            "knn"
            if bool(getattr(args, "student_geometry_only", False))
            else "fully_connected"
        )
    os.environ["DXF_EMPTY_EDGE_FALLBACK"] = str(args.empty_edge_fallback)
    os.environ["DXF_EMPTY_EDGE_K"] = str(int(args.empty_edge_knn_k))
    os.environ["DXF_EPS_SCALE"] = str(float(getattr(args, "dxf_eps_scale", 0.001)))

    keypoints_token = (
        str(getattr(args, "dxf_enhanced_keypoints", "auto") or "auto").strip().lower()
    )
    if keypoints_token == "auto":
        dxf_enhanced_keypoints = bool(getattr(args, "student_geometry_only", False))
    else:
        dxf_enhanced_keypoints = keypoints_token == "true"
    os.environ["DXF_ENHANCED_KEYPOINTS"] = "true" if dxf_enhanced_keypoints else "false"
    args.dxf_enhanced_keypoints = dxf_enhanced_keypoints

    if getattr(args, "dxf_edge_augment_knn_k", None) is None:
        if bool(getattr(args, "student_geometry_only", False)):
            args.dxf_edge_augment_knn_k = 0 if dxf_enhanced_keypoints else 8
        else:
            args.dxf_edge_augment_knn_k = 0
    os.environ["DXF_EDGE_AUGMENT_KNN_K"] = str(int(args.dxf_edge_augment_knn_k))
    strategy_token = (
        str(getattr(args, "dxf_edge_augment_strategy", "auto") or "auto")
        .strip()
        .lower()
    )
    dxf_edge_augment_strategy = (
        "union_all" if strategy_token == "auto" else strategy_token
    )
    os.environ["DXF_EDGE_AUGMENT_STRATEGY"] = dxf_edge_augment_strategy
    args.dxf_edge_augment_strategy = dxf_edge_augment_strategy
    if bool(getattr(args, "student_geometry_only", False)):
        os.environ["DXF_STRIP_TEXT_ENTITIES"] = "true"
        if getattr(args, "dxf_frame_priority_ratio", None) is None:
            args.dxf_frame_priority_ratio = 0.1
        if getattr(args, "dxf_long_line_ratio", None) is None:
            args.dxf_long_line_ratio = 0.4

    # 1) Manifest
    _run(
        [
            python,
            "scripts/build_dxf_label_manifest.py",
            "--input-dir",
            str(dxf_dir),
            "--recursive",
            "--label-mode",
            "filename",
            "--output-csv",
            str(manifest_csv),
        ]
    )

    # 2) Filter weak labels
    filter_stats = _filter_manifest_by_confidence(
        manifest_csv,
        manifest_filtered_csv,
        min_confidence=float(args.min_label_confidence),
    )
    print("manifest_filter=", json.dumps(filter_stats, ensure_ascii=False))

    # 2b) Optional label normalization / cleaning (coarse buckets)
    manifest_for_training = manifest_filtered_csv
    if bool(args.normalize_labels):
        _run(
            [
                python,
                "scripts/normalize_dxf_label_manifest.py",
                "--input-csv",
                str(manifest_for_training),
                "--output-csv",
                str(manifest_normalized_csv),
                "--default-label",
                str(args.normalize_default_label),
            ]
        )
        manifest_for_training = manifest_normalized_csv

    if int(args.clean_min_count) > 0:
        clean_cmd = [
            python,
            "scripts/clean_dxf_label_manifest.py",
            "--input-csv",
            str(manifest_for_training),
            "--output-csv",
            str(manifest_cleaned_csv),
            "--min-count",
            str(int(args.clean_min_count)),
            "--other-label",
            str(args.normalize_default_label),
        ]
        if bool(args.clean_drop_low):
            clean_cmd.append("--drop-low")
        _run(clean_cmd)
        manifest_for_training = manifest_cleaned_csv

    # 3) Train
    train_cmd = _build_train_cmd(
        python=python,
        manifest_csv=Path(manifest_for_training),
        dxf_dir=dxf_dir,
        checkpoint_path=checkpoint_path,
        args=args,
    )
    _run(train_cmd)

    # 4) Eval
    eval_cmd = [
        python,
        "scripts/eval_2d_graph.py",
        "--manifest",
        str(manifest_for_training),
        "--dxf-dir",
        str(dxf_dir),
        "--checkpoint",
        str(checkpoint_path),
        "--batch-size",
        str(int(args.batch_size)),
        "--seed",
        str(int(args.seed)),
        "--output-metrics",
        str(eval_metrics_csv),
        "--output-errors",
        str(eval_errors_csv),
        "--dxf-max-nodes",
        str(int(args.dxf_max_nodes)),
        "--dxf-sampling-strategy",
        str(args.dxf_sampling_strategy),
        "--dxf-sampling-seed",
        str(int(args.dxf_sampling_seed)),
        "--dxf-text-priority-ratio",
        str(float(args.dxf_text_priority_ratio)),
    ]
    if getattr(args, "dxf_frame_priority_ratio", None) is not None:
        eval_cmd.extend(
            [
                "--dxf-frame-priority-ratio",
                str(float(args.dxf_frame_priority_ratio)),
            ]
        )
    if getattr(args, "dxf_long_line_ratio", None) is not None:
        eval_cmd.extend(
            [
                "--dxf-long-line-ratio",
                str(float(args.dxf_long_line_ratio)),
            ]
        )
    _run(eval_cmd)

    # 5) Diagnose (score against weak labels from filename)
    diagnose_cmd = _build_diagnose_cmd(
        python=python,
        dxf_dir=dxf_dir,
        checkpoint_path=checkpoint_path,
        manifest_csv=Path(manifest_for_training),
        out_dir=diagnose_dir,
        args=args,
    )
    _run(diagnose_cmd)

    # Summarize
    summary: Dict[str, Any] = {
        "status": "ok",
        "work_dir": str(work_dir),
        "dxf_dir": str(dxf_dir),
        "checkpoint": str(checkpoint_path),
        "training_profile": str(getattr(args, "training_profile", "none")),
        "training": {
            "model": str(args.model),
            "node_dim": int(getattr(args, "node_dim", 19)),
            "hidden_dim": int(args.hidden_dim),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "loss": str(args.loss),
            "class_weighting": str(args.class_weighting),
            "sampler": str(args.sampler),
        },
        "graph_build": {
            "empty_edge_fallback": str(args.empty_edge_fallback),
            "empty_edge_knn_k": int(args.empty_edge_knn_k),
            "student_geometry_only": bool(args.student_geometry_only),
            "dxf_frame_priority_ratio": (
                float(args.dxf_frame_priority_ratio)
                if args.dxf_frame_priority_ratio is not None
                else None
            ),
            "dxf_long_line_ratio": (
                float(args.dxf_long_line_ratio)
                if args.dxf_long_line_ratio is not None
                else None
            ),
            "dxf_edge_augment_knn_k": int(args.dxf_edge_augment_knn_k),
            "dxf_edge_augment_strategy": str(
                getattr(args, "dxf_edge_augment_strategy", "")
            ),
            "dxf_eps_scale": float(getattr(args, "dxf_eps_scale", 0.001)),
            "dxf_enhanced_keypoints": bool(
                getattr(args, "dxf_enhanced_keypoints", False)
            ),
            "cache": str(args.graph_cache),
            "cache_max_items": int(args.graph_cache_max_items),
            "cache_dir": str(os.getenv("DXF_MANIFEST_DATASET_CACHE_DIR", "")),
        },
        "distillation": {
            "enabled": bool(args.distill),
            "teacher": str(args.teacher),
            "alpha": float(args.distill_alpha),
            "temperature": float(args.distill_temp),
            "mask_filename": _resolve_distill_mask(
                str(args.teacher), str(args.distill_mask_filename)
            ),
        },
        "manifest": {
            "raw": str(manifest_csv),
            "filtered": str(manifest_filtered_csv),
            "normalized": str(manifest_normalized_csv) if args.normalize_labels else "",
            "cleaned": str(manifest_cleaned_csv) if args.clean_min_count > 0 else "",
            "manifest_for_training": str(manifest_for_training),
            **filter_stats,
        },
        "eval": {
            "metrics_csv": str(eval_metrics_csv),
            "errors_csv": str(eval_errors_csv),
        },
        "diagnose": {
            "summary_json": str(diagnose_summary_json),
            "no_text_no_filename": bool(args.diagnose_no_text_no_filename),
        },
    }
    summary_path = work_dir / "pipeline_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
