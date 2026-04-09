#!/usr/bin/env python3
"""Evaluate history-sequence classifier on labeled `.h5` files."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.filename_classifier import FilenameClassifier  # noqa: E402
from src.ml.history_sequence_classifier import HistorySequenceClassifier  # noqa: E402
from src.ml.history_sequence_tools import (  # noqa: E402
    iter_h5_files,
    load_h5_label_pairs_from_manifest,
    macro_f1,
)
from src.core.classification.coarse_labels import normalize_coarse_label  # noqa: E402

_SIDE_SUFFIX_RE = re.compile(r"_\d+$")


def _normalize_h5_stem_for_filename(stem: str) -> str:
    text = str(stem or "").strip()
    if not text:
        return text
    return _SIDE_SUFFIX_RE.sub("", text)


def _resolve_label_from_h5_name(
    path: Path,
    *,
    classifier: FilenameClassifier,
    min_confidence: float,
) -> Optional[str]:
    stem = _normalize_h5_stem_for_filename(path.stem)
    pseudo_name = f"{stem}.dxf"
    prediction = classifier.predict(pseudo_name)
    if prediction.get("label") and float(prediction.get("confidence", 0.0)) >= float(
        min_confidence
    ):
        return str(prediction["label"])
    return None


def _collect_labeled_h5(
    *,
    manifest_path: Optional[Path],
    h5_dir: Optional[Path],
    label_source: str,
    manifest_h5_col: str,
    manifest_label_col: str,
    synonyms_path: Optional[str],
    filename_min_conf: float,
    max_files: int,
    seed: int,
    recursive: bool,
) -> List[Tuple[Path, str]]:
    if label_source == "manifest":
        if manifest_path is None:
            raise ValueError("--label-source=manifest requires --manifest")
        pairs = load_h5_label_pairs_from_manifest(
            manifest_path,
            h5_col=manifest_h5_col,
            label_col=manifest_label_col,
        )
    elif label_source == "filename":
        if h5_dir is None:
            raise ValueError("--label-source=filename requires --h5-dir")
        classifier = FilenameClassifier(synonyms_path=synonyms_path)
        pairs = []
        for h5_path in iter_h5_files(h5_dir, recursive=recursive):
            label = _resolve_label_from_h5_name(
                h5_path,
                classifier=classifier,
                min_confidence=filename_min_conf,
            )
            if label:
                pairs.append((h5_path, label))
    else:  # pragma: no cover
        raise ValueError(f"Unsupported label_source: {label_source}")

    random.Random(int(seed)).shuffle(pairs)
    return pairs[:max_files] if max_files > 0 else pairs


def _default_output_dir() -> Path:
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return Path("reports") / "experiments" / date_str / "history_sequence_eval"


def _normalized_coarse_label(label: str) -> str:
    coarse = normalize_coarse_label(label)
    cleaned = str(label or "").strip()
    return str(coarse or cleaned)


def _top_mismatches(
    expected_labels: Sequence[str],
    predicted_labels: Sequence[str],
    *,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    counts: Counter[Tuple[str, str]] = Counter()
    for expected, predicted in zip(expected_labels, predicted_labels):
        expected_text = str(expected or "").strip()
        predicted_text = str(predicted or "").strip()
        if not expected_text or not predicted_text or expected_text == predicted_text:
            continue
        counts[(expected_text, predicted_text)] += 1

    rows: List[Dict[str, Any]] = []
    for (expected, predicted), count in counts.most_common(max(0, int(top_k))):
        rows.append(
            {
                "expected": expected,
                "predicted": predicted,
                "count": int(count),
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="", help="JSON/CSV manifest path")
    parser.add_argument("--h5-dir", default="", help="Directory containing .h5 files")
    parser.add_argument(
        "--label-source",
        choices=["manifest", "filename"],
        default="manifest",
    )
    parser.add_argument("--manifest-h5-col", default="h5_path")
    parser.add_argument("--manifest-label-col", default="label")
    parser.add_argument("--synonyms-path", default="")
    parser.add_argument("--filename-min-conf", type=float, default=0.8)
    parser.add_argument("--prototypes-path", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--prototype-token-weight", type=float, default=1.0)
    parser.add_argument("--prototype-bigram-weight", type=float, default=1.0)
    parser.add_argument("--min-seq-len", type=int, default=4)
    parser.add_argument("--vec-key", default="vec")
    parser.add_argument("--command-col", type=int, default=0)
    parser.add_argument("--low-conf-threshold", type=float, default=0.5)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest_path = Path(args.manifest).expanduser() if args.manifest else None
    h5_dir = Path(args.h5_dir).expanduser() if args.h5_dir else None
    output_dir = (
        Path(args.output_dir).expanduser() if args.output_dir else _default_output_dir()
    )

    pairs = _collect_labeled_h5(
        manifest_path=manifest_path,
        h5_dir=h5_dir,
        label_source=str(args.label_source),
        manifest_h5_col=str(args.manifest_h5_col),
        manifest_label_col=str(args.manifest_label_col),
        synonyms_path=(str(args.synonyms_path).strip() or None),
        filename_min_conf=float(args.filename_min_conf),
        max_files=max(0, int(args.max_files)),
        seed=int(args.seed),
        recursive=not bool(args.no_recursive),
    )
    if not pairs:
        print(
            json.dumps(
                {"status": "no_samples", "label_source": str(args.label_source)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    classifier = HistorySequenceClassifier(
        prototypes_path=(str(args.prototypes_path).strip() or None),
        model_path=(str(args.model_path).strip() or None),
        min_sequence_length=max(1, int(args.min_seq_len)),
        vec_key=str(args.vec_key),
        command_col=int(args.command_col),
        prototype_token_weight=float(args.prototype_token_weight),
        prototype_bigram_weight=float(args.prototype_bigram_weight),
    )

    rows: List[Dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    expected_all: List[str] = []
    predicted_all: List[str] = []
    expected_ok: List[str] = []
    predicted_ok: List[str] = []
    expected_coarse_all: List[str] = []
    predicted_coarse_all: List[str] = []
    expected_coarse_ok: List[str] = []
    predicted_coarse_ok: List[str] = []
    low_conf_count = 0

    for h5_path, expected_label in pairs:
        payload = classifier.predict_from_h5_file(str(h5_path))
        status = str(payload.get("status") or "")
        pred_label = str(payload.get("label") or "")
        expected_coarse = _normalized_coarse_label(str(expected_label))
        pred_coarse = _normalized_coarse_label(pred_label)
        conf = float(payload.get("confidence", 0.0) or 0.0)
        source = str(payload.get("source") or "")
        status_counts[status or "unknown"] += 1
        if conf < float(args.low_conf_threshold):
            low_conf_count += 1

        expected_all.append(str(expected_label))
        predicted_all.append(pred_label)
        expected_coarse_all.append(expected_coarse)
        predicted_coarse_all.append(pred_coarse)
        if status == "ok":
            expected_ok.append(str(expected_label))
            predicted_ok.append(pred_label)
            expected_coarse_ok.append(expected_coarse)
            predicted_coarse_ok.append(pred_coarse)

        rows.append(
            {
                "h5_path": str(h5_path),
                "expected_label": str(expected_label),
                "expected_coarse_label": expected_coarse,
                "predicted_label": pred_label,
                "predicted_coarse_label": pred_coarse,
                "status": status,
                "confidence": round(conf, 6),
                "source": source,
                "sequence_length": int(payload.get("sequence_length") or 0),
                "unique_commands": int(payload.get("unique_commands") or 0),
                "ok": "Y" if (status == "ok" and pred_label == str(expected_label)) else "N",
                "coarse_ok": (
                    "Y"
                    if (status == "ok" and pred_coarse == expected_coarse)
                    else "N"
                ),
            }
        )

    total = len(rows)
    ok_count = int(status_counts.get("ok", 0))
    ok_correct = sum(
        1
        for row in rows
        if row["status"] == "ok" and row["predicted_label"] == row["expected_label"]
    )
    overall_correct = sum(
        1 for row in rows if row["predicted_label"] == row["expected_label"]
    )
    coarse_ok_correct = sum(
        1
        for row in rows
        if row["status"] == "ok"
        and row["predicted_coarse_label"] == row["expected_coarse_label"]
    )
    coarse_overall_correct = sum(
        1
        for row in rows
        if row["predicted_coarse_label"] == row["expected_coarse_label"]
    )

    summary = {
        "total": total,
        "ok_count": ok_count,
        "coverage": round(ok_count / total, 6) if total else 0.0,
        "accuracy_on_ok": round(ok_correct / ok_count, 6) if ok_count else 0.0,
        "accuracy_overall": round(overall_correct / total, 6) if total else 0.0,
        "coarse_accuracy_on_ok": (
            round(coarse_ok_correct / ok_count, 6) if ok_count else 0.0
        ),
        "coarse_accuracy_overall": (
            round(coarse_overall_correct / total, 6) if total else 0.0
        ),
        "macro_f1_on_ok": (
            round(macro_f1(expected_ok, predicted_ok), 6) if ok_count else 0.0
        ),
        "macro_f1_overall": round(macro_f1(expected_all, predicted_all), 6),
        "coarse_macro_f1_on_ok": (
            round(macro_f1(expected_coarse_ok, predicted_coarse_ok), 6)
            if ok_count
            else 0.0
        ),
        "coarse_macro_f1_overall": round(
            macro_f1(expected_coarse_all, predicted_coarse_all), 6
        ),
        "low_conf_threshold": float(args.low_conf_threshold),
        "low_conf_rate": round(low_conf_count / total, 6) if total else 0.0,
        "status_counts": dict(status_counts),
        "exact_top_mismatches": _top_mismatches(expected_all, predicted_all),
        "coarse_top_mismatches": _top_mismatches(
            expected_coarse_all,
            predicted_coarse_all,
        ),
        "label_source": str(args.label_source),
        "prototypes_path": str(args.prototypes_path or ""),
        "model_path": str(args.model_path or ""),
        "prototype_token_weight": float(args.prototype_token_weight),
        "prototype_bigram_weight": float(args.prototype_bigram_weight),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "results_csv": str(csv_path),
                **summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
