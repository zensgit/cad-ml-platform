#!/usr/bin/env python3
from __future__ import annotations

"""
Generate a standalone HTML report for 2D dedup batch outputs.

Input is a directory produced by:
  scripts/dedup_2d_batch_search_report.py

Expected files:
  - summary.json
  - groups.json
  - matches.csv
  - (optional) precision_diffs/  (when --save-precision-diffs was used)

The HTML report references images via relative paths, so it works offline
when opened locally in a browser.
"""

import argparse
import csv
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MatchRow:
    query_hash: str
    query_file_name: str
    query_path: str
    candidate_hash: str
    candidate_file_name: str
    candidate_path: str
    similarity: float
    visual_similarity: Optional[float]
    precision_score: Optional[float]
    verdict: str
    match_level: int


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _rel(from_dir: Path, target: str) -> str:
    try:
        raw = Path(target)
        if raw.is_absolute():
            abs_path = raw
        else:
            # Prefer packaged/self-contained report paths first (relative to report dir).
            from_dir_candidate = (from_dir / raw).resolve()
            if from_dir_candidate.exists():
                abs_path = from_dir_candidate
            else:
                # Fall back to repository-root relative paths, which are commonly
                # written by batch scripts (e.g., "data/train_artifacts/xxx.png").
                root_candidate = (ROOT / raw).resolve()
                abs_path = root_candidate if root_candidate.exists() else from_dir_candidate
        return os.path.relpath(str(abs_path), str(from_dir))
    except Exception:
        return target


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_matches(matches_csv: Path) -> List[MatchRow]:
    rows: List[MatchRow] = []
    with matches_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                MatchRow(
                    query_hash=str(row.get("query_hash") or ""),
                    query_file_name=str(row.get("query_file_name") or ""),
                    query_path=str(row.get("query_path") or ""),
                    candidate_hash=str(row.get("candidate_hash") or ""),
                    candidate_file_name=str(row.get("candidate_file_name") or ""),
                    candidate_path=str(row.get("candidate_path") or ""),
                    similarity=float(row.get("similarity") or 0.0),
                    visual_similarity=_safe_float(row.get("visual_similarity")),
                    precision_score=_safe_float(row.get("precision_score")),
                    verdict=str(row.get("verdict") or ""),
                    match_level=int(row.get("match_level") or 0),
                )
            )
    # stable sort: highest similarity first
    rows.sort(key=lambda x: float(x.similarity), reverse=True)
    return rows


def _diff_path(precision_diffs_dir: Optional[Path], query_hash: str, candidate_hash: str) -> Optional[Path]:
    if precision_diffs_dir is None:
        return None
    p = precision_diffs_dir / f"{query_hash}__{candidate_hash}.json"
    return p if p.exists() else None


def _render_summary(summary: Dict[str, Any]) -> str:
    return f"""
    <section class="card">
      <h2>Summary</h2>
      <div class="grid">
        <div><span class="k">items_total</span><span class="v">{html.escape(str(summary.get("items_total")))}</span></div>
        <div><span class="k">queries_ok</span><span class="v">{html.escape(str(summary.get("queries_ok")))}</span></div>
        <div><span class="k">queries_failed</span><span class="v">{html.escape(str(summary.get("queries_failed")))}</span></div>
        <div><span class="k">group_rule</span><span class="v">{html.escape(str(summary.get("group_rule")))}</span></div>
        <div><span class="k">group_threshold</span><span class="v">{html.escape(str(summary.get("group_threshold")))}</span></div>
        <div><span class="k">edges</span><span class="v">{html.escape(str(summary.get("edges")))}</span></div>
        <div><span class="k">groups</span><span class="v">{html.escape(str(summary.get("groups")))}</span></div>
        <div><span class="k">matches_rows</span><span class="v">{html.escape(str(summary.get("matches_rows")))}</span></div>
        <div><span class="k">input_dir</span><span class="v mono">{html.escape(str(summary.get("input_dir")))}</span></div>
      </div>
    </section>
    """


def _render_groups(report_dir: Path, groups: Sequence[Dict[str, Any]]) -> str:
    # Sort by size desc, then id
    groups_sorted = sorted(groups, key=lambda g: (-int(g.get("size") or 0), int(g.get("group_id") or 0)))
    blocks: List[str] = []
    for g in groups_sorted:
        gid = int(g.get("group_id") or 0)
        size = int(g.get("size") or 0)
        members = list(g.get("members") or [])

        member_cards: List[str] = []
        for m in members:
            path = str(m.get("path") or "")
            img_src = html.escape(_rel(report_dir, path)) if path else ""
            name = html.escape(Path(path).name) if path else ""
            member_cards.append(
                f"""
                <div class="thumb">
                  <a href="{img_src}" target="_blank" rel="noopener">
                    <img src="{img_src}" alt="{name}" loading="lazy" />
                  </a>
                  <div class="caption mono" title="{name}">{name}</div>
                </div>
                """
            )

        blocks.append(
            f"""
            <section class="card group">
              <h3>Group #{gid} <span class="muted">(size={size})</span></h3>
              <div class="thumbs">{''.join(member_cards)}</div>
            </section>
            """
        )
    return f"""
    <section class="card">
      <h2>Groups</h2>
      <p class="muted">Open images in a new tab to zoom; groups are built from the batch report output.</p>
      {''.join(blocks) if blocks else '<p class="muted">No groups (maybe singletons were excluded).</p>'}
    </section>
    """


def _render_matches(
    report_dir: Path,
    matches: Sequence[MatchRow],
    *,
    precision_diffs_dir: Optional[Path],
    max_rows: int,
) -> str:
    rows: List[str] = []
    for i, m in enumerate(matches[: max(0, int(max_rows))]):
        q_img = html.escape(_rel(report_dir, m.query_path)) if m.query_path else ""
        c_img = html.escape(_rel(report_dir, m.candidate_path)) if m.candidate_path else ""
        q_name = html.escape(m.query_file_name)
        c_name = html.escape(m.candidate_file_name)

        diff_path = _diff_path(precision_diffs_dir, m.query_hash, m.candidate_hash)
        diff_link = ""
        if diff_path is not None:
            diff_href = html.escape(_rel(report_dir, str(diff_path)))
            diff_link = f'<a class="mono" href="{diff_href}" target="_blank" rel="noopener">diff</a>'

        rows.append(
            "<tr>"
            f"<td class='mono'>{i+1}</td>"
            f"<td><a href='{q_img}' target='_blank' rel='noopener'>{q_name}</a></td>"
            f"<td><a href='{c_img}' target='_blank' rel='noopener'>{c_name}</a></td>"
            f"<td class='mono'>{m.similarity:.4f}</td>"
            f"<td class='mono'>{'' if m.visual_similarity is None else f'{m.visual_similarity:.4f}'}</td>"
            f"<td class='mono'>{'' if m.precision_score is None else f'{m.precision_score:.4f}'}</td>"
            f"<td class='mono'>{html.escape(m.verdict)}</td>"
            f"<td class='mono'>{m.match_level}</td>"
            f"<td>{diff_link}</td>"
            "</tr>"
        )

    return f"""
    <section class="card">
      <h2>Matches <span class="muted">(top {min(len(matches), int(max_rows))} rows)</span></h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Query</th>
              <th>Candidate</th>
              <th>Similarity</th>
              <th>Visual</th>
              <th>Precision</th>
              <th>Verdict</th>
              <th>Level</th>
              <th>Diff</th>
            </tr>
          </thead>
	          <tbody>
	            {''.join(rows) if rows else '<tr><td colspan="9" class="muted">No matches</td></tr>'}
	          </tbody>
        </table>
      </div>
    </section>
    """


def generate_html(
    *,
    report_dir: Path,
    summary: Dict[str, Any],
    groups: Sequence[Dict[str, Any]],
    matches: Sequence[MatchRow],
    precision_diffs_dir: Optional[Path],
    max_matches_rows: int,
) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CAD 2D 查重报告</title>
  <style>
    :root {{
      --bg: #0b1020;
      --card: #121a33;
      --text: #e7ecff;
      --muted: #9aa6d6;
      --border: rgba(255,255,255,0.10);
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    }}
    body {{
      margin: 0;
      font-family: var(--sans);
      background: radial-gradient(1200px 800px at 20% 0%, #111b3d 0%, var(--bg) 50%, #060a14 100%);
      color: var(--text);
    }}
    header {{
      padding: 20px 22px 10px;
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      backdrop-filter: blur(10px);
      background: rgba(11,16,32,0.65);
      z-index: 10;
    }}
    header h1 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0.2px;
    }}
    header .sub {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }}
    main {{
      padding: 18px 22px 60px;
      max-width: 1400px;
      margin: 0 auto;
    }}
    .card {{
      background: rgba(18,26,51,0.78);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px 16px;
      margin: 14px 0;
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }}
    h2 {{ margin: 0 0 10px; font-size: 14px; }}
    h3 {{ margin: 0 0 10px; font-size: 13px; }}
    .muted {{ color: var(--muted); }}
    .mono {{ font-family: var(--mono); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .grid > div {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 10px;
      background: rgba(0,0,0,0.10);
      display: flex;
      justify-content: space-between;
      gap: 10px;
    }}
    .k {{ color: var(--muted); font-size: 12px; }}
    .v {{ font-size: 12px; }}
    .thumbs {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 10px;
    }}
    .thumb {{
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      background: rgba(0,0,0,0.12);
    }}
    .thumb img {{
      width: 100%;
      height: 140px;
      object-fit: contain;
      background: rgba(255,255,255,0.03);
      display: block;
    }}
    .caption {{
      padding: 8px 10px;
      font-size: 11px;
      color: var(--text);
      opacity: 0.92;
      border-top: 1px solid var(--border);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    a {{ color: #b8c3ff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      min-width: 900px;
    }}
    thead th {{
      text-align: left;
      color: var(--muted);
      font-weight: 600;
      border-bottom: 1px solid var(--border);
      padding: 10px 8px;
    }}
    tbody td {{
      border-bottom: 1px solid rgba(255,255,255,0.06);
      padding: 8px 8px;
      vertical-align: top;
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>CAD 2D 查重报告</h1>
    <div class="sub mono">{html.escape(str(report_dir.resolve()))}</div>
  </header>
  <main>
    {_render_summary(summary)}
    {_render_groups(report_dir, groups)}
    {_render_matches(report_dir, matches, precision_diffs_dir=precision_diffs_dir, max_rows=max_matches_rows)}
  </main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a static HTML report for 2D dedup outputs.")
    parser.add_argument("report_dir", type=Path, help="Directory containing summary.json/groups.json/matches.csv")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output HTML path (default: <report_dir>/index.html)",
    )
    parser.add_argument(
        "--max-matches-rows",
        type=int,
        default=300,
        help="Max rows shown in the Matches table (default: %(default)s)",
    )
    args = parser.parse_args()

    report_dir: Path = args.report_dir
    if not report_dir.exists():
        raise SystemExit(f"report_dir not found: {report_dir}")

    summary_json = report_dir / "summary.json"
    groups_json = report_dir / "groups.json"
    matches_csv = report_dir / "matches.csv"
    for p in (summary_json, groups_json, matches_csv):
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    summary = _read_json(summary_json)
    groups = _read_json(groups_json)
    matches = _read_matches(matches_csv)

    precision_diffs_dir: Optional[Path] = None
    # Prefer local report_dir/precision_diffs first (packaged/self-contained reports).
    cand = report_dir / "precision_diffs"
    if cand.exists():
        precision_diffs_dir = cand
    else:
        outputs = (summary or {}).get("outputs") if isinstance(summary, dict) else None
        if isinstance(outputs, dict):
            p = outputs.get("precision_diffs_dir")
            if isinstance(p, str) and p:
                cand2 = Path(p)
                precision_diffs_dir = cand2 if cand2.exists() else None

    out_file = args.output_file or (report_dir / "index.html")
    html_text = generate_html(
        report_dir=report_dir,
        summary=summary if isinstance(summary, dict) else {},
        groups=groups if isinstance(groups, list) else [],
        matches=matches,
        precision_diffs_dir=precision_diffs_dir,
        max_matches_rows=int(args.max_matches_rows),
    )
    out_file.write_text(html_text, encoding="utf-8")
    print(f"[ok] wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
