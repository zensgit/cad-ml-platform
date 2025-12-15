#!/usr/bin/env python3
from __future__ import annotations

"""
Evaluate 2D dedup quality using a "group folder" dataset layout.

Dataset layout (recommended):

  dataset_root/
    group_a/
      a1.png
      a1.v2.json
      a2.png
      a2.v2.json
    group_b/
      b1.png
      b1.v2.json

For each query item, positives are the other items in the same group.
The script can optionally index all items into cad-ml-platform first and
trigger a vision index rebuild (recommended for batch ingestion).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".pdf"}


@dataclass(frozen=True)
class DatasetItem:
    group_id: str
    image_path: Path
    geom_json_path: Path
    file_hash: str


@dataclass(frozen=True)
class QueryMetrics:
    group_id: str
    file_hash: str
    positives: int
    top_k: int
    hits: int
    hit_at_k: int
    precision_at_k: float
    recall_at_k: float
    mrr: float


def _iter_images(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _group_id_for_path(dataset_root: Path, image_path: Path) -> str:
    rel = image_path.relative_to(dataset_root)
    parts = rel.parts
    return parts[0] if len(parts) > 1 else "root"


def load_dataset(dataset_root: Path, *, require_json: bool = True) -> List[DatasetItem]:
    items: List[DatasetItem] = []
    for img in _iter_images(dataset_root):
        geom = _find_geom_json(img)
        if geom is None:
            if require_json:
                raise SystemExit(f"Missing geom_json for {img}")
            continue
        items.append(
            DatasetItem(
                group_id=_group_id_for_path(dataset_root, img),
                image_path=img,
                geom_json_path=geom,
                file_hash=_sha256_file(img),
            )
        )
    if not items:
        raise SystemExit(f"No dataset items found under: {dataset_root}")
    return items


def _ranked_matches(response: Dict[str, object]) -> List[Tuple[str, float]]:
    matches: List[Dict[str, object]] = []
    for key in ("duplicates", "similar"):
        block = response.get(key)
        if isinstance(block, list):
            matches.extend([m for m in block if isinstance(m, dict)])
    ranked = sorted(matches, key=lambda m: float(m.get("similarity") or 0.0), reverse=True)
    out: List[Tuple[str, float]] = []
    for m in ranked:
        fh = str(m.get("file_hash") or "")
        if not fh:
            continue
        out.append((fh, float(m.get("similarity") or 0.0)))
    return out


def index_dataset(
    items: Sequence[DatasetItem],
    *,
    base_url: str,
    api_key: str,
    user_name: str,
    upload_to_s3: bool,
    rebuild_index: bool,
) -> None:
    index_endpoint = base_url.rstrip("/") + "/api/v1/dedup/2d/index/add"
    rebuild_endpoint = base_url.rstrip("/") + "/api/v1/dedup/2d/index/rebuild"
    headers = {"X-API-Key": api_key}
    params = {"user_name": user_name, "upload_to_s3": "true" if upload_to_s3 else "false"}

    ok = 0
    for it in items:
        with open(it.image_path, "rb") as f_img, open(it.geom_json_path, "rb") as f_json:
            files = {
                "file": (it.image_path.name, f_img, "application/octet-stream"),
                "geom_json": (it.geom_json_path.name, f_json, "application/json"),
            }
            resp = requests.post(index_endpoint, headers=headers, params=params, files=files, timeout=300)
        if resp.status_code // 100 != 2:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise SystemExit(f"index failed: {it.image_path} -> {resp.status_code}: {detail}")
        ok += 1

    if rebuild_index and ok > 0:
        resp = requests.post(rebuild_endpoint, headers=headers, timeout=120)
        if resp.status_code // 100 != 2:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise SystemExit(f"index rebuild failed: {resp.status_code}: {detail}")


def eval_queries(
    items: Sequence[DatasetItem],
    *,
    base_url: str,
    api_key: str,
    mode: str,
    top_k: int,
    enable_precision: bool,
    precision_top_n: int,
    precision_visual_weight: float,
    precision_geom_weight: float,
) -> List[QueryMetrics]:
    search_endpoint = base_url.rstrip("/") + "/api/v1/dedup/2d/search"
    headers = {"X-API-Key": api_key}

    group_to_hashes: Dict[str, List[str]] = {}
    for it in items:
        group_to_hashes.setdefault(it.group_id, []).append(it.file_hash)

    results: List[QueryMetrics] = []

    for it in items:
        positives = [h for h in group_to_hashes.get(it.group_id, []) if h != it.file_hash]
        positives_set = set(positives)

        params = {
            "mode": mode,
            "max_results": str(max(50, top_k)),
            "enable_precision": "true" if enable_precision else "false",
            "precision_top_n": str(precision_top_n),
            "precision_visual_weight": str(precision_visual_weight),
            "precision_geom_weight": str(precision_geom_weight),
        }
        files: Dict[str, Tuple[str, object, str]] = {}
        files["file"] = (it.image_path.name, open(it.image_path, "rb"), "application/octet-stream")
        if enable_precision:
            files["geom_json"] = (
                it.geom_json_path.name,
                open(it.geom_json_path, "rb"),
                "application/json",
            )

        try:
            t0 = time.perf_counter()
            resp = requests.post(search_endpoint, headers=headers, params=params, files=files, timeout=300)
            _ = time.perf_counter() - t0
        finally:
            for _, f, _ in files.values():
                try:
                    f.close()  # type: ignore[attr-defined]
                except Exception:
                    pass

        if resp.status_code // 100 != 2:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise SystemExit(f"search failed: {it.image_path} -> {resp.status_code}: {detail}")

        data = resp.json()
        ranked = _ranked_matches(data)

        # Remove self if present and truncate to top_k
        ranked = [(fh, s) for fh, s in ranked if fh != it.file_hash][:top_k]
        hits = sum(1 for fh, _ in ranked if fh in positives_set)
        hit_at_k = 1 if hits > 0 else 0

        # Full recall for this query is relative to all positives in the group.
        denom_pos = len(positives_set)
        recall_at_k = (hits / denom_pos) if denom_pos > 0 else 0.0
        precision_at_k = (hits / top_k) if top_k > 0 else 0.0

        mrr = 0.0
        for idx, (fh, _) in enumerate(ranked, start=1):
            if fh in positives_set:
                mrr = 1.0 / idx
                break

        results.append(
            QueryMetrics(
                group_id=it.group_id,
                file_hash=it.file_hash,
                positives=denom_pos,
                top_k=top_k,
                hits=hits,
                hit_at_k=hit_at_k,
                precision_at_k=float(precision_at_k),
                recall_at_k=float(recall_at_k),
                mrr=float(mrr),
            )
        )

    return results


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate 2D dedup using group-folder datasets.")
    parser.add_argument("dataset_root", type=Path, help="Dataset root directory")
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
    parser.add_argument("--upload-to-s3", action="store_true", help="Pass upload_to_s3=true when indexing")
    parser.add_argument(
        "--index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Index dataset before evaluating (default: %(default)s)",
    )
    parser.add_argument(
        "--rebuild-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trigger vision L1/L2 index rebuild after indexing (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        default="balanced",
        help="Search mode for evaluation (fast|balanced|precise) (default: %(default)s)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for metrics (default: %(default)s)")
    parser.add_argument(
        "--enable-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable L4 precision during evaluation (default: %(default)s)",
    )
    parser.add_argument("--precision-top-n", type=int, default=20)
    parser.add_argument("--precision-visual-weight", type=float, default=0.3)
    parser.add_argument("--precision-geom-weight", type=float, default=0.7)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Write per-query metrics as JSONL to this path",
    )
    args = parser.parse_args()

    dataset_root: Path = args.dataset_root
    if not dataset_root.exists():
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    items = load_dataset(dataset_root, require_json=True)
    if args.index:
        index_dataset(
            items,
            base_url=args.base_url,
            api_key=args.api_key,
            user_name=args.user_name,
            upload_to_s3=args.upload_to_s3,
            rebuild_index=args.rebuild_index,
        )

    metrics = eval_queries(
        items,
        base_url=args.base_url,
        api_key=args.api_key,
        mode=args.mode,
        top_k=int(args.top_k),
        enable_precision=bool(args.enable_precision),
        precision_top_n=int(args.precision_top_n),
        precision_visual_weight=float(args.precision_visual_weight),
        precision_geom_weight=float(args.precision_geom_weight),
    )

    with_pos = [m for m in metrics if m.positives > 0]
    summary = {
        "dataset_root": str(dataset_root),
        "queries_total": len(metrics),
        "queries_with_positives": len(with_pos),
        "top_k": int(args.top_k),
        "enable_precision": bool(args.enable_precision),
        "mode": args.mode,
        "metrics": {
            "hit@k": _mean([m.hit_at_k for m in with_pos]),
            "precision@k": _mean([m.precision_at_k for m in with_pos]),
            "recall@k": _mean([m.recall_at_k for m in with_pos]),
            "mrr": _mean([m.mrr for m in with_pos]),
        },
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for m in metrics:
                f.write(json.dumps(m.__dict__, ensure_ascii=False) + "\n")
        print(f"wrote: {args.output_jsonl}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

