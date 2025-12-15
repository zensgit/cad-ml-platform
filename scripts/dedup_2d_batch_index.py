#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable, Optional

import requests


def _iter_images(root: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".pdf"}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _find_geom_json(image_path: Path) -> Optional[Path]:
    candidates = [
        image_path.with_suffix(".json"),
        image_path.with_name(f"{image_path.stem}.v2.json"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch index 2D drawings into cad-ml-platform (vision index + v2 JSON store)."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing images and optional v2 JSON")
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
        "--require-json",
        action="store_true",
        help="Fail if an image has no matching JSON file",
    )
    parser.add_argument(
        "--rebuild-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trigger vision L1/L2 index rebuild after indexing (default: %(default)s)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"input_dir not found: {input_dir}")

    endpoint = args.base_url.rstrip("/") + "/api/v1/dedup/2d/index/add"
    headers = {"X-API-Key": args.api_key}

    total = 0
    ok = 0
    missing_json = 0
    failed = 0

    for img in _iter_images(input_dir):
        total += 1
        geom = _find_geom_json(img)
        if geom is None:
            missing_json += 1
            if args.require_json:
                raise SystemExit(f"Missing geom_json for {img}")
            print(f"[skip] {img} (no geom_json)")
            continue

        with open(img, "rb") as f_img, open(geom, "rb") as f_json:
            files = {
                "file": (img.name, f_img, "application/octet-stream"),
                "geom_json": (geom.name, f_json, "application/json"),
            }
            params = {
                "user_name": args.user_name,
                "upload_to_s3": "true" if args.upload_to_s3 else "false",
            }
            try:
                resp = requests.post(endpoint, headers=headers, params=params, files=files, timeout=120)
            except Exception as e:
                failed += 1
                print(f"[fail] {img} -> request error: {e}")
                continue

        if resp.status_code // 100 != 2:
            failed += 1
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            print(f"[fail] {img} -> {resp.status_code}: {detail}")
            continue

        ok += 1
        data = resp.json()
        print(f"[ok] {img} -> drawing_id={data.get('drawing_id')} file_hash={data.get('file_hash')}")

    if ok > 0 and args.rebuild_index:
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
        f"done: total_images={total} indexed_ok={ok} missing_json={missing_json} failed={failed}"
    )
    return 0 if failed == 0 and (not args.require_json or missing_json == 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
