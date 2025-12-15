#!/usr/bin/env python3
from __future__ import annotations

"""
Package a 2D dedup batch report into a self-contained, shareable ZIP.

Input: a directory produced by scripts/dedup_2d_batch_search_report.py
  - summary.json
  - groups.json (+ optional groups.csv)
  - matches.csv
  - (optional) precision_diffs/

This script copies referenced images (and optional geom JSON) into the report
directory, rewrites paths in groups/matches, regenerates index.html, then zips.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_existing_path(report_dir: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        (report_dir / p).resolve(),
        (ROOT / p).resolve(),
        (Path.cwd() / p).resolve(),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _copy_with_hash_suffix(src: Path, dst_dir: Path) -> Tuple[Path, str]:
    digest8 = _sha256_file(src)[:8]
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{src.stem}__{digest8}{src.suffix}"
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst, digest8


def _iter_image_paths(groups: List[Dict[str, Any]], matches_csv_path: Path) -> Set[str]:
    out: Set[str] = set()
    for g in groups:
        for m in list(g.get("members") or []):
            p = m.get("path")
            if isinstance(p, str) and p:
                out.add(p)
    with matches_csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            for key in ("query_path", "candidate_path"):
                v = row.get(key)
                if isinstance(v, str) and v:
                    out.add(v)
    return out


def _iter_geom_json_paths(groups: List[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for g in groups:
        for m in list(g.get("members") or []):
            p = m.get("json")
            if isinstance(p, str) and p:
                out.add(p)
    return out


def _rewrite_groups_paths(
    groups: List[Dict[str, Any]],
    *,
    image_path_map: Dict[str, str],
    geom_path_map: Optional[Dict[str, str]],
) -> List[Dict[str, Any]]:
    rewritten: List[Dict[str, Any]] = []
    for g in groups:
        gg = dict(g)
        members = []
        for m in list(g.get("members") or []):
            mm = dict(m)
            raw_img = mm.get("path")
            if isinstance(raw_img, str) and raw_img in image_path_map:
                mm["path"] = image_path_map[raw_img]
            raw_json = mm.get("json")
            if geom_path_map is not None and isinstance(raw_json, str) and raw_json in geom_path_map:
                mm["json"] = geom_path_map[raw_json]
            members.append(mm)
        gg["members"] = members
        rewritten.append(gg)
    return rewritten


def _rewrite_groups_csv(groups_csv_in: Path, groups_csv_out: Path, *, image_path_map: Dict[str, str]) -> None:
    with groups_csv_in.open("r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise SystemExit("groups.csv has no header")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    with groups_csv_out.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            p = row.get("path")
            if isinstance(p, str) and p in image_path_map:
                row["path"] = image_path_map[p]
            writer.writerow(row)


def _rewrite_matches_csv(
    matches_csv_in: Path,
    matches_csv_out: Path,
    *,
    image_path_map: Dict[str, str],
) -> None:
    with matches_csv_in.open("r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise SystemExit("matches.csv has no header")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    with matches_csv_out.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for key in ("query_path", "candidate_path"):
                p = row.get(key)
                if isinstance(p, str) and p in image_path_map:
                    row[key] = image_path_map[p]
            writer.writerow(row)


def _zip_dir(dir_path: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    base = dir_path.parent
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(dir_path.rglob("*")):
            if not p.is_file():
                continue
            arcname = os.path.relpath(str(p), str(base))
            zf.write(str(p), arcname)


def main() -> int:
    parser = argparse.ArgumentParser(description="Package a 2D dedup report directory into a ZIP.")
    parser.add_argument("report_dir", type=Path, help="Directory produced by dedup_2d_batch_search_report.py")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write packaged report dir (default: <report_dir>_package)",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Output zip path (default: <output_dir>.zip)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_dir if exists")
    parser.add_argument(
        "--include-geom-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy member geom JSON (v2) referenced in groups.json (default: %(default)s)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing referenced files instead of failing",
    )
    args = parser.parse_args()

    report_dir: Path = args.report_dir
    summary_path = report_dir / "summary.json"
    groups_json_path = report_dir / "groups.json"
    matches_csv_path = report_dir / "matches.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary.json: {summary_path}")
    if not groups_json_path.exists():
        raise SystemExit(f"Missing groups.json: {groups_json_path}")
    if not matches_csv_path.exists():
        raise SystemExit(f"Missing matches.csv: {matches_csv_path}")

    output_dir = args.output_dir or report_dir.with_name(report_dir.name + "_package")
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"output_dir exists (use --overwrite): {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load/prepare summary (we will rewrite outputs for packaged dir at the end).
    summary = _read_json(summary_path)
    if not isinstance(summary, dict):
        summary = {"_raw_summary": summary}
    precision_diffs_src = report_dir / "precision_diffs"
    if precision_diffs_src.exists() and precision_diffs_src.is_dir():
        shutil.copytree(precision_diffs_src, output_dir / "precision_diffs")

    groups: List[Dict[str, Any]] = _read_json(groups_json_path)
    if not isinstance(groups, list):
        raise SystemExit("groups.json is not a list")

    # Copy images & build path map.
    images_dir = output_dir / "assets" / "images"
    image_path_map: Dict[str, str] = {}
    missing: List[str] = []
    for raw in sorted(_iter_image_paths(groups, matches_csv_path)):
        try:
            src = _resolve_existing_path(report_dir, raw)
            if not src.exists():
                raise FileNotFoundError(str(src))
            dst, _ = _copy_with_hash_suffix(src, images_dir)
            image_path_map[raw] = os.path.relpath(str(dst), str(output_dir))
        except Exception:
            if args.allow_missing:
                missing.append(raw)
                continue
            raise

    geom_path_map: Optional[Dict[str, str]] = None
    if args.include_geom_json:
        geoms_dir = output_dir / "assets" / "geoms"
        geom_path_map = {}
        for raw in sorted(_iter_geom_json_paths(groups)):
            try:
                src = _resolve_existing_path(report_dir, raw)
                if not src.exists():
                    raise FileNotFoundError(str(src))
                dst, _ = _copy_with_hash_suffix(src, geoms_dir)
                geom_path_map[raw] = os.path.relpath(str(dst), str(output_dir))
            except Exception:
                if args.allow_missing:
                    missing.append(raw)
                    continue
                raise

    groups_rewritten = _rewrite_groups_paths(groups, image_path_map=image_path_map, geom_path_map=geom_path_map)
    _write_json(output_dir / "groups.json", groups_rewritten)

    # Rewrite CSVs.
    groups_csv_in = report_dir / "groups.csv"
    if groups_csv_in.exists():
        _rewrite_groups_csv(groups_csv_in, output_dir / "groups.csv", image_path_map=image_path_map)
    _rewrite_matches_csv(matches_csv_path, output_dir / "matches.csv", image_path_map=image_path_map)

    # Rewrite summary outputs for packaged location (best-effort).
    try:
        if isinstance(summary, dict):
            summary["packaged"] = True
            summary["packaged_from"] = str(report_dir)
            outputs = summary.get("outputs")
            if not isinstance(outputs, dict):
                outputs = {}
                summary["outputs"] = outputs
            outputs["matches_csv"] = str(output_dir / "matches.csv")
            outputs["groups_json"] = str(output_dir / "groups.json")
            groups_csv_out = output_dir / "groups.csv"
            outputs["groups_csv"] = str(groups_csv_out) if groups_csv_out.exists() else None
            outputs["summary_json"] = str(output_dir / "summary.json")
            precision_diffs_out = output_dir / "precision_diffs"
            outputs["precision_diffs_dir"] = (
                str(precision_diffs_out) if precision_diffs_out.exists() else None
            )
    except Exception:
        pass
    _write_json(output_dir / "summary.json", summary)

    # Re-generate HTML with rewritten paths.
    from scripts.dedup_2d_generate_html_report import main as gen_html_main  # type: ignore

    argv0 = sys.argv[0]
    try:
        sys.argv = [
            argv0,
            str(output_dir),
            "--max-matches-rows",
            "300",
        ]
        rc = gen_html_main()
        if rc != 0:
            raise SystemExit(f"HTML generation failed: rc={rc}")
    finally:
        sys.argv = [argv0]

    # Zip packaged dir.
    zip_path = args.zip_path or output_dir.with_suffix(".zip")
    _zip_dir(output_dir, zip_path)

    if missing:
        print(f"[warn] missing referenced files: {len(missing)}")
    print(f"[ok] packaged_dir={output_dir}")
    print(f"[ok] zip={zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
