#!/usr/bin/env python3
"""
Plot evaluation trends from reports/eval_history/*.json.

Generates line charts for combined, OCR, and history-sequence evaluation signals.

Usage:
    python3 scripts/eval_trend.py --out reports/eval_history/plots
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure matplotlib uses a non-interactive backend and writable config dir
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_mpl_cfg_dir = ROOT / "reports" / "eval_history" / "mplcache"
_mpl_cfg_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cfg_dir))

try:
    import matplotlib  # type: ignore
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    print("ERROR: matplotlib is required. Install with 'pip install matplotlib'.")
    raise
HISTORY_DIR = ROOT / "reports" / "eval_history"

from scripts import summarize_eval_signal_runs as eval_signal_canonical
from scripts import summarize_history_sequence_runs as history_canonical


def _normalize_history_file(path_text: object, *, root_dir: Path) -> str:
    text = str(path_text or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = root_dir / path
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def load_history(
    history_dir: Path | None = None,
    *,
    eval_signal_summary_json: Path | None = None,
    eval_signal_summary: dict | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    root_dir = history_dir or HISTORY_DIR
    if not root_dir.exists():
        return [], [], []

    if isinstance(eval_signal_summary, dict):
        signal_rows = eval_signal_canonical._rows_from_summary(eval_signal_summary)
    else:
        signal_summary = eval_signal_canonical._load_or_build_summary(
            eval_signal_summary_json or (root_dir / "eval_signal_experiment_summary.json"),
            eval_history_dir=root_dir,
            report_glob="*.json",
        )
        signal_rows = eval_signal_canonical._rows_from_summary(signal_summary)

    combined = [
        dict(row)
        for row in signal_rows
        if str(row.get("report_type") or "").strip() == "combined"
    ]
    for row in combined:
        row["_file"] = _normalize_history_file(row.get("report_path"), root_dir=root_dir)

    ocr_only = [
        dict(row)
        for row in signal_rows
        if str(row.get("report_type") or "").strip() == "ocr"
    ]
    for row in ocr_only:
        row["_file"] = _normalize_history_file(row.get("report_path"), root_dir=root_dir)

    return combined, ocr_only, []


def load_history_sequence_rows(
    history_dir: Path | None = None,
    *,
    history_sequence_summary_json: Path | None = None,
    history_sequence_summary: dict | None = None,
) -> list[dict]:
    if isinstance(history_sequence_summary, dict):
        return history_canonical._rows_from_summary(history_sequence_summary)

    root_dir = history_dir or HISTORY_DIR
    summary_json = history_sequence_summary_json or (
        root_dir / "history_sequence_experiment_summary.json"
    )
    history_summary = history_canonical._load_or_build_summary(
        summary_json,
        eval_history_dir=root_dir,
        report_glob="*.json",
    )
    return history_canonical._rows_from_summary(history_summary)


def to_dt(ts: str) -> datetime:
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return datetime.fromtimestamp(0)


def plot_combined(combined: list[dict], outdir: Path) -> Path | None:
    if not combined:
        return None
    xs = [to_dt(d.get("timestamp", "1970-01-01T00:00:00Z")) for d in combined]
    ys = [d.get("combined", {}).get("combined_score", 0.0) for d in combined]
    vws = [d.get("combined", {}).get("vision_weight", 0.5) for d in combined]

    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, marker="o", label="Combined Score")
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.title("Combined Evaluation Trend")
    plt.xlabel("Time")
    plt.ylabel("Score (0-1)")
    # Annotate last point with weights
    if xs:
        plt.annotate(f"w_v={vws[-1]:.2f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(8, 8))

    out = outdir / "combined_trend.png"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def plot_ocr_only(ocr_only: list[dict], outdir: Path) -> Path | None:
    if not ocr_only:
        return None
    xs = [to_dt(d.get("timestamp", "1970-01-01T00:00:00Z")) for d in ocr_only]
    dr = [d.get("metrics", {}).get("dimension_recall", 0.0) for d in ocr_only]
    bs = [d.get("metrics", {}).get("brier_score", 1.0) for d in ocr_only]
    one_minus_brier = [max(0.0, 1.0 - b) for b in bs]

    plt.figure(figsize=(8, 4))
    plt.plot(xs, dr, marker="o", label="Dimension Recall")
    plt.plot(xs, one_minus_brier, marker="s", label="1 - Brier")
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.title("OCR Metrics Trend")
    plt.xlabel("Time")
    plt.ylabel("Score (0-1)")
    plt.legend()

    out = outdir / "ocr_trend.png"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def _history_named_summary(payload: dict) -> dict:
    block = payload.get("named_command_summary")
    return block if isinstance(block, dict) else {}


def _history_named_error_summary(payload: dict) -> dict:
    block = payload.get("named_command_error_summary")
    return block if isinstance(block, dict) else {}


def plot_history_sequence(history_rows: list[dict], outdir: Path) -> Path | None:
    if not history_rows:
        return None
    xs = [history_canonical._parse_ts(row.get("timestamp")) for row in history_rows]
    history_metrics = [
        row.get("history_metrics") if isinstance(row.get("history_metrics"), dict) else {}
        for row in history_rows
    ]
    coverage = [_safe_float(m.get("coverage"), 0.0) for m in history_metrics]
    accuracy = [_safe_float(m.get("accuracy_overall"), 0.0) for m in history_metrics]
    macro_f1 = [_safe_float(m.get("macro_f1_overall"), 0.0) for m in history_metrics]
    explainability = [
        _safe_float(
            row.get("named_command_summary", {}).get("named_command_explainability_rate"), 0.0
        )
        for row in history_rows
    ]
    error_rate = [
        _safe_float(row.get("named_command_error_summary", {}).get("overall_incorrect_rate"), 0.0)
        for row in history_rows
    ]
    low_conf_rate = [
        _safe_float(row.get("named_command_error_summary", {}).get("overall_low_conf_rate"), 0.0)
        for row in history_rows
    ]

    plt.figure(figsize=(9, 4.5))
    plt.plot(xs, coverage, marker="o", label="Coverage")
    plt.plot(xs, accuracy, marker="s", label="Accuracy")
    plt.plot(xs, macro_f1, marker="^", label="Macro F1")
    plt.plot(xs, explainability, marker="D", label="Named Explainability")
    plt.plot(xs, error_rate, marker="x", label="Named Error Rate")
    plt.plot(xs, low_conf_rate, marker="P", label="Named Low-Conf Rate")
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.title("History Sequence Metrics Trend")
    plt.xlabel("Time")
    plt.ylabel("Score (0-1)")
    plt.legend()

    latest_summary = (
        history_rows[-1].get("named_command_summary")
        if isinstance(history_rows[-1].get("named_command_summary"), dict)
        else {}
    )
    latest_error_summary = (
        history_rows[-1].get("named_command_error_summary")
        if isinstance(history_rows[-1].get("named_command_error_summary"), dict)
        else {}
    )
    latest_worst_family = (
        latest_error_summary.get("worst_primary_family")
        if isinstance(latest_error_summary.get("worst_primary_family"), dict)
        else {}
    )
    latest_surface = str(latest_summary.get("sequence_surface_kind") or "n/a")
    latest_vocab = str(latest_summary.get("named_command_vocabulary_kind") or "n/a")
    latest_worst_family_value = str(latest_worst_family.get("value") or "n/a")
    plt.annotate(
        f"{latest_surface}\n{latest_vocab}\nworst_family={latest_worst_family_value}",
        (xs[-1], coverage[-1]),
        textcoords="offset points",
        xytext=(8, 8),
    )

    out = outdir / "history_sequence_trend.png"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def plot_history_sequence_by_surface(history_rows: list[dict], outdir: Path) -> Path | None:
    if not history_rows:
        return None
    groups = history_canonical._group_rows_by_surface(history_rows)
    if not groups:
        return None
    ordered_keys = sorted(groups)
    fig, axes = plt.subplots(
        len(ordered_keys),
        1,
        figsize=(9, max(4.5, 3.6 * len(ordered_keys))),
        squeeze=False,
    )
    for idx, surface_key in enumerate(ordered_keys):
        axis = axes[idx][0]
        group = groups[surface_key]
        xs = [history_canonical._parse_ts(row.get("timestamp")) for row in group]
        accuracy = [
            _safe_float(row.get("history_metrics", {}).get("accuracy_overall"), 0.0)
            for row in group
        ]
        macro_f1 = [
            _safe_float(row.get("history_metrics", {}).get("macro_f1_overall"), 0.0)
            for row in group
        ]
        explainability = [
            _safe_float(
                row.get("named_command_summary", {}).get("named_command_explainability_rate"),
                0.0,
            )
            for row in group
        ]
        error_rate = [
            _safe_float(
                row.get("named_command_error_summary", {}).get("overall_incorrect_rate"),
                0.0,
            )
            for row in group
        ]
        axis.plot(xs, accuracy, marker="s", label="Accuracy")
        axis.plot(xs, macro_f1, marker="^", label="Macro F1")
        axis.plot(xs, explainability, marker="D", label="Named Explainability")
        axis.plot(xs, error_rate, marker="x", label="Named Error Rate")
        axis.set_ylim(0, 1)
        axis.grid(True, linestyle=":", alpha=0.5)
        axis.set_title(f"Surface: {surface_key} ({len(group)} reports)")
        axis.set_xlabel("Time")
        axis.set_ylabel("Score (0-1)")
        axis.legend(loc="lower right")
    fig.suptitle("History Sequence Trend By Exact Surface", y=0.995)
    out = outdir / "history_sequence_surface_trend.png"
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def write_history_sequence_metadata(history_sequence: list[dict], outdir: Path) -> Path | None:
    if not history_sequence:
        return None
    window = history_canonical._build_window_summary(history_sequence)
    latest = max(history_sequence, key=lambda row: history_canonical._parse_ts(row.get("timestamp")))
    latest_summary = (
        latest.get("named_command_summary") if isinstance(latest.get("named_command_summary"), dict) else {}
    )
    payload = {
        "report_count": int(window["report_count"]),
        "latest_timestamp": str(latest.get("timestamp") or ""),
        "latest_sequence_surface_kind": str(window["latest_sequence_surface_kind"]),
        "latest_named_command_vocabulary_kind": str(window["latest_named_vocabulary_kind"]),
        "latest_named_command_authoritative_names_known": bool(
            latest_summary.get("named_command_authoritative_names_known", False)
        ),
        "latest_prediction_surface_counts": dict(
            latest_summary.get("prediction_surface_counts") or {}
        ),
        "latest_named_command_explainability_rate": _safe_float(
            latest_summary.get("named_command_explainability_rate"),
            0.0,
        ),
        "latest_named_command_error_rate": _safe_float(
            latest.get("named_command_error_summary", {}).get("overall_incorrect_rate"), 0.0
        ),
        "latest_named_command_low_conf_rate": _safe_float(
            latest.get("named_command_error_summary", {}).get("overall_low_conf_rate"), 0.0
        ),
        "latest_named_command_low_conf_threshold": _safe_float(
            latest.get("named_command_error_summary", {}).get("low_conf_threshold"), 0.0
        ),
        "latest_worst_primary_family": str(window["latest_worst_primary_family"]),
        "latest_worst_primary_reference_surface": str(
            window["latest_worst_primary_reference_surface"]
        ),
        "latest_worst_primary_status": str(window["latest_worst_primary_status"]),
    }
    out = outdir / "history_sequence_trend_metadata.json"
    outdir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_history_sequence_surface_metadata(history_rows: list[dict], outdir: Path) -> Path | None:
    if not history_rows:
        return None
    window = history_canonical._build_window_summary(history_rows)
    payload = {
        "surface_kind": "history_sequence_surface_trend_metadata",
        "report_count": int(len(history_rows)),
        "surface_count": int(window["surface_group_count"]),
        "best_surface_key_by_mean_accuracy_overall": str(
            window["best_surface_key_by_mean_accuracy_overall"]
        ),
        "surface_groups": list(window["surface_groups"]),
    }
    out = outdir / "history_sequence_surface_trend_metadata.json"
    outdir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot evaluation trends from history")
    parser.add_argument("--out", type=str, default=str(HISTORY_DIR / "plots"), help="Output directory for plots")
    parser.add_argument(
        "--eval-history-dir",
        type=str,
        default=str(HISTORY_DIR),
        help="Evaluation history directory.",
    )
    parser.add_argument(
        "--eval-signal-summary-json",
        type=str,
        default="",
        help=(
            "Optional canonical eval-signal experiment summary JSON. "
            "When present, combined/OCR trend generation prefers the canonical artifact."
        ),
    )
    parser.add_argument(
        "--history-sequence-summary-json",
        type=str,
        default="",
        help=(
            "Optional canonical history-sequence experiment summary JSON. "
            "When present, history-sequence trend generation prefers the canonical artifact."
        ),
    )
    args = parser.parse_args(argv)

    history_dir = Path(args.eval_history_dir)
    signal_summary_json = (
        Path(str(args.eval_signal_summary_json))
        if str(args.eval_signal_summary_json).strip()
        else None
    )
    combined, ocr_only, history_sequence = load_history(
        history_dir,
        eval_signal_summary_json=signal_summary_json,
    )
    summary_json = (
        Path(str(args.history_sequence_summary_json))
        if str(args.history_sequence_summary_json).strip()
        else history_dir / "history_sequence_experiment_summary.json"
    )
    history_rows = load_history_sequence_rows(
        history_dir,
        history_sequence_summary_json=summary_json,
    )
    outdir = Path(args.out)

    cpath = plot_combined(combined, outdir)
    opath = plot_ocr_only(ocr_only, outdir)
    hpath = plot_history_sequence(history_rows, outdir)
    hmeta = write_history_sequence_metadata(history_rows, outdir)
    hs_path = plot_history_sequence_by_surface(history_rows, outdir)
    hs_meta = write_history_sequence_surface_metadata(history_rows, outdir)

    if cpath:
        print(f"Combined trend saved: {cpath}")
    else:
        print("No combined history found.")

    if opath:
        print(f"OCR trend saved: {opath}")
    else:
        print("No OCR-only history found.")

    if hpath:
        print(f"History sequence trend saved: {hpath}")
    else:
        print("No history-sequence history found.")

    if hmeta:
        print(f"History sequence trend metadata saved: {hmeta}")
    if hs_path:
        print(f"History sequence surface trend saved: {hs_path}")
    if hs_meta:
        print(f"History sequence surface trend metadata saved: {hs_meta}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
