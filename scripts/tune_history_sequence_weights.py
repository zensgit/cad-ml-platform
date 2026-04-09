#!/usr/bin/env python3
"""Grid-search token and bigram weights for history-sequence prototypes."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
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
        raise ValueError(f"unsupported label_source: {label_source}")

    random.Random(int(seed)).shuffle(pairs)
    return pairs[:max_files] if max_files > 0 else pairs


def _parse_float_grid(raw: str, *, default: Sequence[float]) -> List[float]:
    text = str(raw or "").strip()
    if not text:
        return [float(v) for v in default]
    values: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        return [float(v) for v in default]
    deduped: List[float] = []
    seen: set[float] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _evaluate_with_weights(
    *,
    pairs: Sequence[Tuple[Path, str]],
    prototypes_path: str,
    model_path: str,
    min_seq_len: int,
    vec_key: str,
    command_col: int,
    token_weight: float,
    bigram_weight: float,
    low_conf_threshold: float,
) -> Dict[str, Any]:
    classifier = HistorySequenceClassifier(
        prototypes_path=prototypes_path or None,
        model_path=model_path or None,
        min_sequence_length=max(1, int(min_seq_len)),
        vec_key=str(vec_key),
        command_col=int(command_col),
        prototype_token_weight=float(token_weight),
        prototype_bigram_weight=float(bigram_weight),
    )

    expected_all: List[str] = []
    predicted_all: List[str] = []
    expected_ok: List[str] = []
    predicted_ok: List[str] = []
    low_conf_count = 0
    ok_count = 0
    ok_correct = 0
    overall_correct = 0

    for h5_path, expected_label in pairs:
        payload = classifier.predict_from_h5_file(str(h5_path))
        status = str(payload.get("status") or "")
        pred_label = str(payload.get("label") or "")
        conf = float(payload.get("confidence", 0.0) or 0.0)
        if conf < float(low_conf_threshold):
            low_conf_count += 1

        expected_all.append(str(expected_label))
        predicted_all.append(pred_label)
        if status == "ok":
            ok_count += 1
            expected_ok.append(str(expected_label))
            predicted_ok.append(pred_label)
            if pred_label == str(expected_label):
                ok_correct += 1
        if pred_label == str(expected_label):
            overall_correct += 1

    total = len(pairs)
    return {
        "token_weight": float(token_weight),
        "bigram_weight": float(bigram_weight),
        "total": total,
        "ok_count": ok_count,
        "coverage": (ok_count / total) if total else 0.0,
        "accuracy_on_ok": (ok_correct / ok_count) if ok_count else 0.0,
        "accuracy_overall": (overall_correct / total) if total else 0.0,
        "macro_f1_on_ok": macro_f1(expected_ok, predicted_ok) if ok_count else 0.0,
        "macro_f1_overall": macro_f1(expected_all, predicted_all),
        "low_conf_rate": (low_conf_count / total) if total else 0.0,
    }


def _default_output_dir() -> Path:
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return Path("reports") / "experiments" / date_str / "history_sequence_tuning"


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
    parser.add_argument("--prototypes-path", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--token-weight-grid", default="0.5,1.0,1.5")
    parser.add_argument("--bigram-weight-grid", default="0.0,0.5,1.0,1.5")
    parser.add_argument(
        "--objective",
        choices=[
            "accuracy_overall",
            "macro_f1_overall",
            "accuracy_on_ok",
            "macro_f1_on_ok",
        ],
        default="macro_f1_overall",
    )
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
    prototypes_path = str(Path(args.prototypes_path).expanduser())

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

    token_grid = _parse_float_grid(str(args.token_weight_grid), default=[1.0])
    bigram_grid = _parse_float_grid(str(args.bigram_weight_grid), default=[1.0])

    rows: List[Dict[str, Any]] = []
    for token_weight in token_grid:
        for bigram_weight in bigram_grid:
            rows.append(
                _evaluate_with_weights(
                    pairs=pairs,
                    prototypes_path=prototypes_path,
                    model_path=str(args.model_path or ""),
                    min_seq_len=int(args.min_seq_len),
                    vec_key=str(args.vec_key),
                    command_col=int(args.command_col),
                    token_weight=float(token_weight),
                    bigram_weight=float(bigram_weight),
                    low_conf_threshold=float(args.low_conf_threshold),
                )
            )

    objective = str(args.objective)
    best_row = max(
        rows,
        key=lambda item: (
            float(item.get(objective, 0.0)),
            -float(item["low_conf_rate"]),
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "weight_grid.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_payload = {
        "objective": objective,
        "best": best_row,
        "token_grid": token_grid,
        "bigram_grid": bigram_grid,
        "total_samples": len(pairs),
        "label_source": str(args.label_source),
        "prototypes_path": prototypes_path,
        "model_path": str(args.model_path or ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "weight_grid_csv": str(csv_path),
        },
        "recommended_env": {
            "HISTORY_SEQUENCE_PROTOTYPE_TOKEN_WEIGHT": str(best_row["token_weight"]),
            "HISTORY_SEQUENCE_PROTOTYPE_BIGRAM_WEIGHT": str(best_row["bigram_weight"]),
        },
    }

    env_path = output_dir / "recommended_history_sequence.env"
    env_lines = [
        "# Auto-generated by scripts/tune_history_sequence_weights.py",
        f"HISTORY_SEQUENCE_PROTOTYPE_TOKEN_WEIGHT={best_row['token_weight']}",
        f"HISTORY_SEQUENCE_PROTOTYPE_BIGRAM_WEIGHT={best_row['bigram_weight']}",
        f"HISTORY_SEQUENCE_PROTOTYPES_PATH={prototypes_path}",
    ]
    env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    best_payload["artifacts"]["recommended_env_file"] = str(env_path)

    best_path = output_dir / "best_config.json"
    best_path.write_text(
        json.dumps(best_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "best_config": str(best_path),
                "weight_grid_csv": str(csv_path),
                **best_payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
