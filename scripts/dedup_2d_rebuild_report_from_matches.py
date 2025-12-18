#!/usr/bin/env python3
from __future__ import annotations

"""
Rebuild a 2D dedup batch report (groups/verdicts/summary) from an existing matches.csv.

Why:
  - Local L4 full-matrix scoring is O(N^2) and can be expensive.
  - Once you have a full similarity matrix, you should be able to tune thresholds
    and regenerate reports without recomputing scores.

Inputs:
  - source_report_dir/matches.csv (required)
  - source_report_dir/summary.json (optional; used to carry forward input_dir)

Outputs (output_dir):
  - matches.csv (verdict re-labeled with new thresholds)
  - groups.json / groups.csv (rebuilt with new group rule/threshold)
  - summary.json
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class MatchRow:
    query_hash: str
    query_path: str
    query_file_name: str
    candidate_hash: str
    candidate_path: str
    candidate_file_name: str
    candidate_drawing_id: str
    similarity: float
    visual_similarity: str
    precision_score: str
    verdict: str
    match_level: int


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent: Dict[str, str] = {x: x for x in items}
        self.rank: Dict[str, int] = {x: 0 for x in items}

    def find(self, x: str) -> str:
        p = self.parent.get(x, x)
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent.get(x, x)

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank.get(ra, 0) < self.rank.get(rb, 0):
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank.get(ra, 0) == self.rank.get(rb, 0):
            self.rank[ra] = self.rank.get(ra, 0) + 1


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _read_matches(matches_csv: Path) -> List[MatchRow]:
    rows: List[MatchRow] = []
    with matches_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                MatchRow(
                    query_hash=str(row.get("query_hash") or ""),
                    query_path=str(row.get("query_path") or ""),
                    query_file_name=str(row.get("query_file_name") or ""),
                    candidate_hash=str(row.get("candidate_hash") or ""),
                    candidate_path=str(row.get("candidate_path") or ""),
                    candidate_file_name=str(row.get("candidate_file_name") or ""),
                    candidate_drawing_id=str(row.get("candidate_drawing_id") or ""),
                    similarity=_safe_float(row.get("similarity")),
                    visual_similarity=str(row.get("visual_similarity") or ""),
                    precision_score=str(row.get("precision_score") or ""),
                    verdict=str(row.get("verdict") or ""),
                    match_level=int(row.get("match_level") or 0),
                )
            )
    return rows


def _determine_verdict(similarity: float, *, duplicate_threshold: float, similar_threshold: float) -> str:
    if similarity >= duplicate_threshold:
        return "duplicate"
    if similarity >= similar_threshold:
        return "similar"
    return "different"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild groups/verdicts/summary from an existing matches.csv without recomputing scores."
    )
    parser.add_argument("source_report_dir", type=Path, help="Directory containing matches.csv (and optional summary.json)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory to write rebuilt report files",
    )
    parser.add_argument(
        "--preset",
        choices=["strict", "version", "loose"],
        default=None,
        help="Convenience preset for thresholds + group rule (default: %(default)s)",
    )
    parser.add_argument(
        "--group-rule",
        choices=["verdict", "threshold"],
        default="verdict",
        help="How to link edges for clustering (default: %(default)s)",
    )
    parser.add_argument(
        "--group-threshold",
        type=float,
        default=0.95,
        help="Used when --group-rule=threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.95,
        help="Threshold for labeling duplicate (default: %(default)s)",
    )
    parser.add_argument(
        "--similar-threshold",
        type=float,
        default=0.80,
        help="Threshold for labeling similar (default: %(default)s)",
    )
    parser.add_argument(
        "--include-singletons",
        action="store_true",
        help="Include singleton groups in groups outputs",
    )
    args = parser.parse_args()

    source_report_dir: Path = args.source_report_dir
    matches_csv = source_report_dir / "matches.csv"
    if not matches_csv.exists():
        raise SystemExit(f"matches.csv not found: {matches_csv}")

    # Apply preset defaults (only when caller did not override from defaults).
    presets = {
        "strict": {
            "duplicate_threshold": 0.95,
            "similar_threshold": 0.80,
            "group_rule": "verdict",
            "group_threshold": 0.95,
        },
        "version": {
            "duplicate_threshold": 0.95,
            "similar_threshold": 0.70,
            "group_rule": "threshold",
            "group_threshold": 0.70,
        },
        "loose": {
            "duplicate_threshold": 0.90,
            "similar_threshold": 0.50,
            "group_rule": "threshold",
            "group_threshold": 0.50,
        },
    }
    if args.preset is not None:
        preset = presets[str(args.preset)]
        if float(args.duplicate_threshold) == 0.95:
            args.duplicate_threshold = float(preset["duplicate_threshold"])
        if float(args.similar_threshold) == 0.80:
            args.similar_threshold = float(preset["similar_threshold"])
        if args.group_rule == "verdict" and float(args.group_threshold) == 0.95:
            args.group_rule = str(preset["group_rule"])
            args.group_threshold = float(preset["group_threshold"])

    dup_th = float(args.duplicate_threshold)
    sim_th = float(args.similar_threshold)
    if not (0.0 <= sim_th <= dup_th <= 1.0):
        raise SystemExit("Invalid thresholds: require 0 <= similar_threshold <= duplicate_threshold <= 1")

    rows = _read_matches(matches_csv)
    if not rows:
        raise SystemExit(f"No rows found in matches.csv: {matches_csv}")

    # Determine dataset items
    items: Dict[str, Dict[str, str]] = {}
    for r in rows:
        if r.query_hash and r.query_hash not in items:
            items[r.query_hash] = {"file_hash": r.query_hash, "file_name": r.query_file_name, "path": r.query_path}
        if r.candidate_hash and r.candidate_hash not in items:
            items[r.candidate_hash] = {
                "file_hash": r.candidate_hash,
                "file_name": r.candidate_file_name,
                "path": r.candidate_path,
            }

    input_hashes: Set[str] = set(items.keys())
    edges: Set[Tuple[str, str]] = set()

    # Relabel verdicts and build edges
    relabeled: List[MatchRow] = []
    for r in rows:
        if not r.query_hash or not r.candidate_hash:
            continue
        verdict = _determine_verdict(r.similarity, duplicate_threshold=dup_th, similar_threshold=sim_th)
        relabeled.append(
            MatchRow(
                query_hash=r.query_hash,
                query_path=r.query_path,
                query_file_name=r.query_file_name,
                candidate_hash=r.candidate_hash,
                candidate_path=r.candidate_path,
                candidate_file_name=r.candidate_file_name,
                candidate_drawing_id=r.candidate_drawing_id,
                similarity=r.similarity,
                visual_similarity=r.visual_similarity,
                precision_score=r.precision_score,
                verdict=verdict,
                match_level=r.match_level,
            )
        )

        should_link = False
        if str(args.group_rule) == "verdict":
            should_link = verdict == "duplicate"
        else:
            should_link = r.similarity >= float(args.group_threshold)
        if should_link and r.query_hash in input_hashes and r.candidate_hash in input_hashes:
            a, b = (r.query_hash, r.candidate_hash)
            if a != b:
                if a > b:
                    a, b = b, a
                edges.add((a, b))

    uf = _UnionFind(input_hashes)
    for a, b in edges:
        uf.union(a, b)

    root_to_members: Dict[str, List[str]] = {}
    for h in sorted(input_hashes):
        root_to_members.setdefault(uf.find(h), []).append(h)

    groups: List[Dict[str, Any]] = []
    for idx, (root, members) in enumerate(sorted(root_to_members.items()), start=1):
        if (not args.include_singletons) and len(members) <= 1:
            continue
        groups.append(
            {
                "group_id": idx,
                "root": root,
                "size": len(members),
                "members": [items[m] for m in members if m in items],
            }
        )

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_matches = out_dir / "matches.csv"
    out_groups_json = out_dir / "groups.json"
    out_groups_csv = out_dir / "groups.csv"
    out_summary = out_dir / "summary.json"

    with out_matches.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "query_hash",
                "query_path",
                "query_file_name",
                "candidate_hash",
                "candidate_path",
                "candidate_file_name",
                "candidate_drawing_id",
                "similarity",
                "visual_similarity",
                "precision_score",
                "verdict",
                "match_level",
            ],
        )
        w.writeheader()
        for r in relabeled:
            w.writerow(
                {
                    "query_hash": r.query_hash,
                    "query_path": r.query_path,
                    "query_file_name": r.query_file_name,
                    "candidate_hash": r.candidate_hash,
                    "candidate_path": r.candidate_path,
                    "candidate_file_name": r.candidate_file_name,
                    "candidate_drawing_id": r.candidate_drawing_id,
                    "similarity": f"{r.similarity:.10f}",
                    "visual_similarity": r.visual_similarity,
                    "precision_score": r.precision_score,
                    "verdict": r.verdict,
                    "match_level": str(r.match_level),
                }
            )

    out_groups_json.write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_groups_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group_id", "size", "file_hash", "file_name", "path"])
        w.writeheader()
        for g in groups:
            gid = int(g.get("group_id") or 0)
            size = int(g.get("size") or 0)
            for m in g.get("members") or []:
                w.writerow(
                    {
                        "group_id": gid,
                        "size": size,
                        "file_hash": str(m.get("file_hash") or ""),
                        "file_name": str(m.get("file_name") or ""),
                        "path": str(m.get("path") or ""),
                    }
                )

    input_dir_value = ""
    source_summary = source_report_dir / "summary.json"
    if source_summary.exists():
        try:
            s = _read_json(source_summary)
            input_dir_value = str(s.get("input_dir") or "")
        except Exception:
            input_dir_value = ""

    summary_obj = {
        "input_dir": input_dir_value,
        "items_total": len(input_hashes),
        "queries_ok": len(input_hashes),
        "queries_failed": 0,
        "within_input_only": True,
        "group_rule": str(args.group_rule),
        "group_threshold": float(args.group_threshold),
        "edges": len(edges),
        "groups": len(groups),
        "matches_rows": len(relabeled),
        "outputs": {
            "matches_csv": str(out_matches),
            "groups_json": str(out_groups_json),
            "groups_csv": str(out_groups_csv),
            "summary_json": str(out_summary),
            "responses_jsonl": None,
            "diff_images_dir": None,
            "precision_diffs_dir": None,
        },
        "rebuilt_from": str(source_report_dir),
        "thresholds": {"duplicate_threshold": dup_th, "similar_threshold": sim_th},
    }
    out_summary.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "source_report_dir": str(source_report_dir),
                "output_dir": str(out_dir),
                "items_total": len(input_hashes),
                "matches_rows": len(relabeled),
                "edges": len(edges),
                "groups": len(groups),
                "group_rule": str(args.group_rule),
                "group_threshold": float(args.group_threshold),
                "duplicate_threshold": dup_th,
                "similar_threshold": sim_th,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

