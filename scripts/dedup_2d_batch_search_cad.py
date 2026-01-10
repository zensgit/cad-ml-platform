#!/usr/bin/env python3
from __future__ import annotations

"""
Batch DWG/DXF -> PNG/JSON -> index -> search for 2D dedup.

This script supports large DWG batches by:
1) Converting DWG -> DXF (optional, uses ODA or custom command)
2) Rendering DXF -> PNG and extracting v2 JSON
3) Indexing PNG (+ geom JSON) into dedupcad-vision via cad-ml-platform
4) Running async search and exporting JSON (+ optional CSV)
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.dedupcad_precision.cad_pipeline import (  # noqa: E402
    DxfRenderConfig,
    OdaConverterConfig,
    convert_dwg_to_dxf_cmd,
    convert_dwg_to_dxf_oda,
    extract_geom_json_from_dxf,
    render_dxf_to_png,
    resolve_oda_exe_from_env,
)


def _iter_cad_files(root: Path) -> Iterable[Path]:
    exts = {".dxf", ".dwg"}
    if root.is_file():
        if root.suffix.lower() in exts:
            yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _maybe_convert_dwg(
    cad_path: Path,
    out_dxf_path: Path,
    *,
    dwg_to_dxf: str,
    oda_exe: Optional[Path],
    oda_output_version: str,
    dwg_to_dxf_cmd: str,
    overwrite: bool,
) -> Path:
    if cad_path.suffix.lower() == ".dxf":
        return cad_path
    if out_dxf_path.exists() and not overwrite:
        return out_dxf_path

    if dwg_to_dxf == "skip":
        raise RuntimeError("dwg_to_dxf=skip but input is .dwg")

    if dwg_to_dxf in {"auto", "oda"}:
        exe = oda_exe or resolve_oda_exe_from_env()
        if exe is not None:
            convert_dwg_to_dxf_oda(
                cad_path,
                out_dxf_path,
                cfg=OdaConverterConfig(exe_path=exe, output_version=oda_output_version),
            )
            return out_dxf_path
        if dwg_to_dxf == "oda":
            raise RuntimeError("Missing ODA converter (use --oda-exe or ODA_FILE_CONVERTER_EXE)")

    if dwg_to_dxf_cmd:
        convert_dwg_to_dxf_cmd(cad_path, out_dxf_path, cmd_template=dwg_to_dxf_cmd)
        return out_dxf_path

    raise RuntimeError(
        "DWG conversion unavailable: set --oda-exe/ODA_FILE_CONVERTER_EXE "
        "or use --dwg-to-dxf cmd with --dwg-to-dxf-cmd (or env DWG_TO_DXF_CMD)"
    )


def _post(
    *,
    url: str,
    headers: Dict[str, str],
    params: Dict[str, Any],
    files: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    resp = requests.post(url, headers=headers, params=params, files=files, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _poll_job(
    *,
    base_url: str,
    job_id: str,
    headers: Dict[str, str],
    poll_interval: float,
    poll_timeout: float,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/v1/dedup/2d/jobs/{job_id}"
    deadline = time.time() + poll_timeout
    last_resp: Dict[str, Any] = {}
    while time.time() < deadline:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        last_resp = data
        status = data.get("status")
        if status in {"completed", "failed", "canceled"}:
            return data
        time.sleep(poll_interval)
    last_resp["status"] = "timeout"
    last_resp["error"] = f"poll_timeout>{poll_timeout}s"
    return last_resp


def _filter_self_matches(result: Dict[str, Any], *, file_hash: str) -> Dict[str, Any]:
    if not file_hash:
        return result
    cleaned = dict(result)
    duplicates = cleaned.get("duplicates") or []
    similar = cleaned.get("similar") or []
    cleaned["duplicates"] = [item for item in duplicates if item.get("file_hash") != file_hash]
    cleaned["similar"] = [item for item in similar if item.get("file_hash") != file_hash]
    cleaned["total_matches"] = len(cleaned["duplicates"]) + len(cleaned["similar"])
    return cleaned


def _summarize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    if not result:
        return summary
    summary["success"] = result.get("success")
    summary["total_matches"] = result.get("total_matches")
    summary["duplicate_count"] = len(result.get("duplicates") or [])
    summary["similar_count"] = len(result.get("similar") or [])
    summary["final_level"] = result.get("final_level")
    timing = result.get("timing") or {}
    summary["timing_total_ms"] = timing.get("total_ms")
    if result.get("error"):
        summary["error"] = result.get("error")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch DWG/DXF -> PNG/JSON -> index -> search for dedup2d."
    )
    parser.add_argument("input_path", type=Path, help="Directory or file containing DWG/DXF")
    parser.add_argument(
        "--base-url",
        default=os.getenv("CAD_ML_PLATFORM_URL", "http://localhost:8000"),
        help="cad-ml-platform base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("CAD_ML_PLATFORM_API_KEY", "test"),
        help="X-API-Key header value (default: %(default)s)",
    )
    parser.add_argument(
        "--user-name",
        default=os.getenv("USER", "batch"),
        help="Indexing user_name query param (default: %(default)s)",
    )
    parser.add_argument("--mode", default="balanced", help="Search mode (default: %(default)s)")
    parser.add_argument("--max-results", type=int, default=5, help="Search max_results (default: %(default)s)")
    parser.add_argument(
        "--compute-diff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable compute_diff during search (default: %(default)s)",
    )
    parser.add_argument(
        "--enable-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable precision verification (default: %(default)s)",
    )
    parser.add_argument(
        "--upload-to-s3",
        action="store_true",
        help="Upload indexed files to S3 storage",
    )
    parser.add_argument(
        "--index-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Index all files before searching (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip indexing step (assumes files already indexed)",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip search step (useful for indexing-only runs)",
    )
    parser.add_argument(
        "--exclude-self",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter out self matches by file_hash (default: %(default)s)",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Process at most N files (0 = no limit)")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index within the discovered file list (default: %(default)s)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate artifacts if present")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data/dedupcad_batch_search"),
        help="Directory to store DXF/PNG/JSON artifacts (default: %(default)s)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/dedup2d_batch_search_results.json"),
        help="JSON output file (default: %(default)s)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output file (omit to skip)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=300.0,
        help="Polling timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=300.0,
        help="HTTP request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--rebuild-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trigger index rebuild after indexing (default: %(default)s)",
    )
    parser.add_argument(
        "--dwg-to-dxf",
        choices=["auto", "oda", "cmd", "skip"],
        default="auto",
        help="How to convert DWG to DXF (default: %(default)s)",
    )
    parser.add_argument(
        "--oda-exe",
        type=Path,
        default=None,
        help="Path to ODAFileConverter (or set env ODA_FILE_CONVERTER_EXE)",
    )
    parser.add_argument(
        "--oda-output-version",
        default=os.getenv("ODA_OUTPUT_VERSION", "ACAD2018"),
        help="ODA output DXF version (default: %(default)s)",
    )
    parser.add_argument(
        "--dwg-to-dxf-cmd",
        default=os.getenv("DWG_TO_DXF_CMD", ""),
        help='Custom command template for DWG->DXF. Use {input} and {output}.',
    )
    parser.add_argument("--render-size-px", type=int, default=1024, help="DXF render size in px")
    parser.add_argument("--render-dpi", type=int, default=200, help="DXF render DPI")
    parser.add_argument(
        "--render-margin-ratio",
        type=float,
        default=0.05,
        help="Margin ratio around extents when rendering",
    )
    args = parser.parse_args()

    input_path = args.input_path
    if not input_path.exists():
        raise SystemExit(f"input_path not found: {input_path}")

    work_dir = args.work_dir
    _ensure_dir(work_dir)
    os.environ.setdefault("DEDUPCAD2_CACHE_DIR", str(work_dir / "dedupcad2_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(work_dir / "mplcache"))

    headers = {"X-API-Key": args.api_key}
    base_url = args.base_url.rstrip("/")
    index_url = f"{base_url}/api/v1/dedup/2d/index/add"
    search_url = f"{base_url}/api/v1/dedup/2d/search"

    render_cfg = DxfRenderConfig(
        size_px=args.render_size_px,
        dpi=args.render_dpi,
        margin_ratio=args.render_margin_ratio,
    )

    results: List[Dict[str, Any]] = []
    cad_files = list(_iter_cad_files(input_path))
    if args.start_index > 0:
        cad_files = cad_files[args.start_index :]
    if args.max_files > 0:
        cad_files = cad_files[: args.max_files]

    indexed: List[Dict[str, Any]] = []
    for cad_path in cad_files:
        rel = cad_path.relative_to(input_path) if input_path.is_dir() else cad_path.name
        out_base = work_dir / Path(rel).with_suffix("")
        _ensure_dir(out_base.parent)

        item: Dict[str, Any] = {"input_path": str(cad_path), "status": "pending"}
        try:
            dxf_path = _maybe_convert_dwg(
                cad_path,
                out_base.with_suffix(".dxf"),
                dwg_to_dxf=args.dwg_to_dxf,
                oda_exe=args.oda_exe,
                oda_output_version=args.oda_output_version,
                dwg_to_dxf_cmd=args.dwg_to_dxf_cmd,
                overwrite=args.overwrite,
            )
            item["dxf_path"] = str(dxf_path)

            json_path = out_base.with_suffix(".v2.json")
            if args.overwrite or not json_path.exists():
                try:
                    geom = extract_geom_json_from_dxf(dxf_path)
                    json_path.write_text(
                        json.dumps(geom, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    item["geom_json_path"] = str(json_path)
                except Exception as e:
                    item["geom_json_error"] = str(e)
            else:
                item["geom_json_path"] = str(json_path)

            png_path = out_base.with_suffix(".png")
            if args.overwrite or not png_path.exists():
                render_dxf_to_png(dxf_path, png_path, config=render_cfg)
            item["png_path"] = str(png_path)
        except Exception as e:
            item["status"] = "failed"
            item["error"] = f"prepare_failed: {e}"
            results.append(item)
            continue

        if args.index_first and not args.skip_index:
            try:
                with open(item["png_path"], "rb") as f_img:
                    files = {"file": (Path(item["png_path"]).name, f_img, "image/png")}
                    geom_path = item.get("geom_json_path")
                    if geom_path:
                        with open(geom_path, "rb") as f_json:
                            files["geom_json"] = (
                                Path(geom_path).name,
                                f_json,
                                "application/json",
                            )
                            params = {"user_name": args.user_name, "upload_to_s3": str(args.upload_to_s3).lower()}
                            index_resp = _post(
                                url=index_url,
                                headers=headers,
                                params=params,
                                files=files,
                                timeout=args.request_timeout,
                            )
                    else:
                        params = {"user_name": args.user_name, "upload_to_s3": str(args.upload_to_s3).lower()}
                        index_resp = _post(
                            url=index_url,
                            headers=headers,
                            params=params,
                            files=files,
                            timeout=args.request_timeout,
                        )
                    item["index"] = index_resp
                    indexed.append(index_resp)
            except Exception as e:
                item["index"] = {"success": False, "error": str(e)}

        results.append(item)

    if args.index_first and args.rebuild_index and indexed and not args.skip_index:
        try:
            resp = requests.post(
                f"{base_url}/api/v1/dedup/2d/index/rebuild",
                headers=headers,
                timeout=args.request_timeout,
            )
            resp.raise_for_status()
        except Exception:
            pass

    if not args.skip_search:
        for item in results:
            if "png_path" not in item:
                continue
            try:
                with open(item["png_path"], "rb") as f_img:
                    files = {"file": (Path(item["png_path"]).name, f_img, "image/png")}
                    geom_path = item.get("geom_json_path")
                    params = {
                        "async": "true",
                        "mode": args.mode,
                        "max_results": args.max_results,
                        "compute_diff": str(args.compute_diff).lower(),
                        "enable_precision": str(args.enable_precision).lower(),
                    }
                    if geom_path:
                        with open(geom_path, "rb") as f_json:
                            files["geom_json"] = (
                                Path(geom_path).name,
                                f_json,
                                "application/json",
                            )
                            submit = _post(
                                url=search_url,
                                headers=headers,
                                params=params,
                                files=files,
                                timeout=args.request_timeout,
                            )
                    else:
                        submit = _post(
                            url=search_url,
                            headers=headers,
                            params=params,
                            files=files,
                            timeout=args.request_timeout,
                        )
                    item["search_submit"] = submit
                    job_id = submit.get("job_id")
                    if not job_id:
                        item["search"] = {"status": "failed", "error": "missing job_id"}
                        continue
                    job = _poll_job(
                        base_url=base_url,
                        job_id=job_id,
                        headers=headers,
                        poll_interval=args.poll_interval,
                        poll_timeout=args.poll_timeout,
                    )
                    if args.exclude_self:
                        file_hash = ""
                        if isinstance(item.get("index"), dict):
                            file_hash = str(item["index"].get("file_hash") or "")
                        if file_hash and isinstance(job.get("result"), dict):
                            job["result"] = _filter_self_matches(job["result"], file_hash=file_hash)
                    item["search"] = job
                    if isinstance(job.get("result"), dict):
                        item["summary"] = _summarize_result(job["result"])
            except Exception as e:
                item["search"] = {"status": "failed", "error": str(e)}

    output_json = args.output_json
    _ensure_dir(output_json.parent)
    output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_csv:
        output_csv = args.output_csv
        _ensure_dir(output_csv.parent)
        fields = [
            "input_path",
            "dxf_path",
            "png_path",
            "geom_json_path",
            "index_success",
            "index_drawing_id",
            "index_file_hash",
            "search_status",
            "search_success",
            "total_matches",
            "duplicate_count",
            "similar_count",
            "final_level",
            "timing_total_ms",
            "error",
        ]
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for item in results:
                index = item.get("index") or {}
                search = item.get("search") or {}
                summary = item.get("summary") or {}
                row = {
                    "input_path": item.get("input_path"),
                    "dxf_path": item.get("dxf_path"),
                    "png_path": item.get("png_path"),
                    "geom_json_path": item.get("geom_json_path"),
                    "index_success": index.get("success"),
                    "index_drawing_id": index.get("drawing_id"),
                    "index_file_hash": index.get("file_hash"),
                    "search_status": search.get("status"),
                    "search_success": summary.get("success"),
                    "total_matches": summary.get("total_matches"),
                    "duplicate_count": summary.get("duplicate_count"),
                    "similar_count": summary.get("similar_count"),
                    "final_level": summary.get("final_level"),
                    "timing_total_ms": summary.get("timing_total_ms"),
                    "error": summary.get("error") or item.get("error") or search.get("error"),
                }
                writer.writerow(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
