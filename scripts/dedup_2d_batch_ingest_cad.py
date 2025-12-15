#!/usr/bin/env python3
from __future__ import annotations

"""
Batch ingest CAD drawings (DXF/DWG) for 2D dedup:

DXF -> (render PNG) + (extract v2 JSON) -> POST /api/v1/dedup/2d/index/add
DWG -> DXF (optional converter) -> same as above
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

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
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _ensure_mpl_cache_dir(dir_path: Path) -> None:
    cache_dir = dir_path / "mplcache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch ingest DXF/DWG into cad-ml-platform (vision index + v2 JSON precision store)."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing DXF/DWG files")
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
    parser.add_argument(
        "--upload-to-s3",
        action="store_true",
        help="Pass upload_to_s3=true to vision index endpoint",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data/dedupcad_batch"),
        help="Directory to store generated DXF/PNG/JSON artifacts (default: %(default)s)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Process at most N CAD files (0 = no limit) (default: %(default)s)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate artifacts even if present")
    parser.add_argument("--dry-run", action="store_true", help="Only generate artifacts; do not call API")
    parser.add_argument(
        "--rebuild-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trigger vision L1/L2 index rebuild after ingest (default: %(default)s)",
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
        help="Path to ODAFileConverter.exe (or set env ODA_FILE_CONVERTER_EXE)",
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

    input_dir: Path = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"input_dir not found: {input_dir}")

    work_dir: Path = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    _ensure_mpl_cache_dir(work_dir)
    os.environ.setdefault("DEDUPCAD2_CACHE_DIR", str(work_dir / "dedupcad2_cache"))

    endpoint = args.base_url.rstrip("/") + "/api/v1/dedup/2d/index/add"
    headers = {"X-API-Key": args.api_key}
    params = {
        "user_name": args.user_name,
        "upload_to_s3": "true" if args.upload_to_s3 else "false",
    }

    render_cfg = DxfRenderConfig(
        size_px=args.render_size_px,
        dpi=args.render_dpi,
        margin_ratio=args.render_margin_ratio,
    )

    total = 0
    ok = 0
    failed = 0
    skipped_dwg = 0
    max_files = int(args.max_files)

    for cad_path in _iter_cad_files(input_dir):
        if max_files > 0 and total >= max_files:
            break
        total += 1
        rel = cad_path.relative_to(input_dir)
        stem = rel.with_suffix("")
        out_base = work_dir / stem
        out_base.parent.mkdir(parents=True, exist_ok=True)

        try:
            if cad_path.suffix.lower() == ".dxf":
                dxf_path = cad_path
            else:
                dxf_path = out_base.with_suffix(".dxf")
                if (not dxf_path.exists()) or args.overwrite:
                    mode = args.dwg_to_dxf
                    if mode == "skip":
                        skipped_dwg += 1
                        print(f"[skip] {cad_path} (dwg_to_dxf=skip)")
                        continue

                    if mode in ("auto", "oda"):
                        oda_exe = args.oda_exe or resolve_oda_exe_from_env()
                        if oda_exe is not None:
                            convert_dwg_to_dxf_oda(
                                cad_path,
                                dxf_path,
                                cfg=OdaConverterConfig(exe_path=oda_exe, output_version=args.oda_output_version),
                            )
                        elif mode == "oda":
                            raise RuntimeError("Missing --oda-exe (or ODA_FILE_CONVERTER_EXE)")
                        else:
                            # fall back to cmd if provided
                            if not args.dwg_to_dxf_cmd:
                                raise RuntimeError(
                                    "DWG conversion unavailable: set --oda-exe/ODA_FILE_CONVERTER_EXE "
                                    "or use --dwg-to-dxf cmd with --dwg-to-dxf-cmd (or env DWG_TO_DXF_CMD)"
                                )
                            convert_dwg_to_dxf_cmd(cad_path, dxf_path, cmd_template=args.dwg_to_dxf_cmd)
                    elif mode == "cmd":
                        if not args.dwg_to_dxf_cmd:
                            raise RuntimeError("Missing --dwg-to-dxf-cmd (or env DWG_TO_DXF_CMD)")
                        convert_dwg_to_dxf_cmd(cad_path, dxf_path, cmd_template=args.dwg_to_dxf_cmd)
                    else:
                        raise RuntimeError(f"Unsupported dwg_to_dxf mode: {mode}")

            json_path = out_base.with_suffix(".v2.json")
            if (not json_path.exists()) or args.overwrite:
                geom = extract_geom_json_from_dxf(dxf_path)
                json_path.write_text(json.dumps(geom, ensure_ascii=False, indent=2), encoding="utf-8")

            png_path = out_base.with_suffix(".png")
            if (not png_path.exists()) or args.overwrite:
                render_dxf_to_png(dxf_path, png_path, config=render_cfg)

            if args.dry_run:
                print(f"[ok] {cad_path} -> generated {png_path.name} + {json_path.name}")
                ok += 1
                continue

            with open(png_path, "rb") as f_img, open(json_path, "rb") as f_json:
                files = {
                    "file": (png_path.name, f_img, "image/png"),
                    "geom_json": (json_path.name, f_json, "application/json"),
                }
                resp = requests.post(endpoint, headers=headers, params=params, files=files, timeout=300)

            if resp.status_code // 100 != 2:
                failed += 1
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text
                print(f"[fail] {cad_path} -> {resp.status_code}: {detail}")
                continue

            data = resp.json()
            ok += 1
            print(
                f"[ok] {cad_path} -> drawing_id={data.get('drawing_id')} file_hash={data.get('file_hash')}"
            )
        except Exception as e:
            failed += 1
            print(f"[fail] {cad_path} -> {e}")

    if not args.dry_run and ok > 0 and args.rebuild_index:
        rebuild_endpoint = args.base_url.rstrip("/") + "/api/v1/dedup/2d/index/rebuild"
        try:
            t0 = time.perf_counter()
            resp = requests.post(rebuild_endpoint, headers=headers, timeout=120)
            dt_ms = (time.perf_counter() - t0) * 1000
            if resp.status_code // 100 != 2:
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text
                print(f"[warn] index rebuild failed: {resp.status_code}: {detail}")
            else:
                print(f"[ok] index rebuilt in {dt_ms:.1f}ms: {resp.json()}")
        except Exception as e:
            print(f"[warn] index rebuild request error: {e}")

    print(
        f"done: total_files={total} ok={ok} failed={failed} skipped_dwg={skipped_dwg} work_dir={work_dir}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
