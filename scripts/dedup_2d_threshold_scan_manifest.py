#!/usr/bin/env python3
from __future__ import annotations

"""
Threshold scan / calibration for 2D dedup scores using a weak-label manifest.

This script is designed for the "training drawings" workflow used in this repo:

1) Generate artifacts (DWG/DXF -> PNG + v2 JSON)
2) Produce a full similarity matrix via local L4:
     python3 scripts/dedup_2d_batch_search_report.py data/train_artifacts \
       --engine local_l4 --top-k 108 --group-rule threshold --group-threshold 0.0 \
       --output-dir data/dedup_report_train_local_full
3) Build weak labels (same base name / version groups), e.g.:
     data/train_drawings_manifest/expected_groups.json
4) Scan thresholds to pick:
   - "strict duplicate" threshold (high precision)
   - "version/similar" threshold (higher recall)
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PairItem:
    score: float
    is_positive: bool
    left_stem: str
    right_stem: str


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    predicted: int
    tp: int
    fp: int
    fn: int
    precision: Optional[float]
    recall: Optional[float]


def _stem(file_name: str) -> str:
    # Matches are usually PNG file names; keep it robust for any extension.
    return Path(file_name).stem


def _load_expected_groups(expected_groups_json: Path) -> Dict[str, int]:
    data = json.loads(expected_groups_json.read_text(encoding="utf-8"))
    mapping: Dict[str, int] = {}
    for g in data:
        gid = int(g["group_id"])
        for m in g.get("members") or []:
            stem = str(m.get("stem") or "").strip()
            if not stem:
                continue
            mapping[stem] = gid
    if not mapping:
        raise SystemExit(f"No members found in expected_groups: {expected_groups_json}")
    return mapping


def _iter_pairs(matches_csv: Path, *, stem_to_group: Dict[str, int]) -> Iterable[PairItem]:
    # Build undirected pair score matrix by taking max score across both directions.
    pair_score: Dict[Tuple[str, str], float] = {}
    with matches_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"query_file_name", "candidate_file_name", "similarity"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"matches.csv missing columns {sorted(missing)}: {matches_csv}")

        for row in reader:
            qstem = _stem(str(row.get("query_file_name") or ""))
            cstem = _stem(str(row.get("candidate_file_name") or ""))
            if not qstem or not cstem or qstem == cstem:
                continue
            if qstem not in stem_to_group or cstem not in stem_to_group:
                continue
            try:
                score = float(row.get("similarity") or 0.0)
            except Exception:
                continue

            a, b = sorted((qstem, cstem))
            key = (a, b)
            prev = pair_score.get(key)
            if prev is None or score > prev:
                pair_score[key] = score

    for (a, b), score in pair_score.items():
        is_pos = stem_to_group[a] == stem_to_group[b]
        yield PairItem(score=score, is_positive=is_pos, left_stem=a, right_stem=b)


def _quantile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    if p <= 0:
        return float(values[0])
    if p >= 1:
        return float(values[-1])
    idx = int(round((len(values) - 1) * p))
    return float(values[idx])


def compute_metrics(pairs: Sequence[PairItem], threshold: float) -> ThresholdMetrics:
    tp = fp = fn = 0
    for it in pairs:
        pred = it.score >= threshold
        if pred and it.is_positive:
            tp += 1
        elif pred and (not it.is_positive):
            fp += 1
        elif (not pred) and it.is_positive:
            fn += 1
    predicted = tp + fp
    precision = tp / predicted if predicted > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    return ThresholdMetrics(
        threshold=float(threshold),
        predicted=predicted,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan similarity thresholds using expected_groups.json + matches.csv (weak-label calibration)."
    )
    parser.add_argument(
        "--expected-groups-json",
        type=Path,
        default=Path("data/train_drawings_manifest/expected_groups.json"),
        help="Weak-label groups manifest (default: %(default)s)",
    )
    parser.add_argument(
        "--matches-csv",
        type=Path,
        default=Path("data/dedup_report_train_local_070/matches.csv"),
        help="Batch report matches.csv containing pairwise similarities (default: %(default)s)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
        help="Thresholds to evaluate (default: %(default)s)",
    )
    parser.add_argument(
        "--top-negatives",
        type=int,
        default=10,
        help="Print top-N negative pairs by score (default: %(default)s)",
    )
    args = parser.parse_args()

    expected_groups_json: Path = args.expected_groups_json
    matches_csv: Path = args.matches_csv
    if not expected_groups_json.exists():
        raise SystemExit(f"expected_groups_json not found: {expected_groups_json}")
    if not matches_csv.exists():
        raise SystemExit(f"matches_csv not found: {matches_csv}")

    stem_to_group = _load_expected_groups(expected_groups_json)
    pairs = list(_iter_pairs(matches_csv, stem_to_group=stem_to_group))
    if not pairs:
        raise SystemExit("No usable pairs found (check that file stems match expected_groups.json)")

    pos_scores = sorted([p.score for p in pairs if p.is_positive])
    neg_scores = sorted([p.score for p in pairs if not p.is_positive])
    print(
        json.dumps(
            {
                "pairs_total": len(pairs),
                "positives": len(pos_scores),
                "negatives": len(neg_scores),
                "pos_min": _quantile(pos_scores, 0.0),
                "pos_p10": _quantile(pos_scores, 0.1),
                "pos_p50": _quantile(pos_scores, 0.5),
                "pos_p90": _quantile(pos_scores, 0.9),
                "pos_max": _quantile(pos_scores, 1.0),
                "neg_p90": _quantile(neg_scores, 0.9),
                "neg_p95": _quantile(neg_scores, 0.95),
                "neg_p99": _quantile(neg_scores, 0.99),
                "neg_max": _quantile(neg_scores, 1.0),
            },
            ensure_ascii=False,
        )
    )

    print("threshold,predicted,tp,fp,fn,precision,recall")
    for t in [float(x) for x in args.thresholds]:
        m = compute_metrics(pairs, t)
        print(
            ",".join(
                [
                    f"{m.threshold:.4f}",
                    str(m.predicted),
                    str(m.tp),
                    str(m.fp),
                    str(m.fn),
                    "" if m.precision is None else f"{m.precision:.6f}",
                    "" if m.recall is None else f"{m.recall:.6f}",
                ]
            )
        )

    if int(args.top_negatives) > 0:
        top_n = int(args.top_negatives)
        neg_pairs = sorted(
            [p for p in pairs if not p.is_positive],
            key=lambda p: p.score,
            reverse=True,
        )[:top_n]
        print("top_negative_pairs:")
        for p in neg_pairs:
            print(f"{p.score:.6f} {p.left_stem} <-> {p.right_stem}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

