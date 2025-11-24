#!/usr/bin/env python
"""Feature vector migration script.

Recomputes feature vectors for stored analyses when upgrading FEATURE_VERSION.

Usage:
  python scripts/migrate_feature_version.py --from v1 --to v2 --batch-size 50 --dry-run 0

Environment:
  FEATURE_VERSION (target version when recomputing)
  VECTOR_STORE_BACKEND (optional faiss) to also add vectors to ANN index
"""

from __future__ import annotations
import argparse
import os
import asyncio
from typing import List

from src.core.feature_extractor import FeatureExtractor
from src.models.cad_document import CadDocument
from src.utils.analysis_metrics import feature_migration_total
from src.core.similarity import register_vector, FaissVectorStore
from src.utils.cache import get_cached_result


async def migrate(ids: List[str], from_version: str, to_version: str, dry_run: bool, batch_size: int):
    extractor = FeatureExtractor()
    backend = os.getenv("VECTOR_STORE_BACKEND", "memory")
    if backend == "faiss":
        faiss_store = FaissVectorStore()
    for i, analysis_id in enumerate(ids, 1):
        # Load cached analysis result
        data = await get_cached_result(f"analysis_result:{analysis_id}")
        if not data:
            feature_migration_total.labels(status="skipped").inc()
            continue
        features = data.get("features", {})
        current_version = features.get("feature_version", "v1")
        if current_version == to_version:
            feature_migration_total.labels(status="skipped").inc()
            continue
        # Build document from cached stats (best-effort)
        stats = data.get("statistics", {})
        bbox = stats.get("bounding_box", {})
        doc = CadDocument(
            file_name=data.get("file_name", "unknown"),
            format=data.get("file_format", "unknown"),
        )
        # minimal reconstruction
        doc.bounding_box.width = bbox.get("width", 0.0)
        doc.bounding_box.height = bbox.get("height", 0.0)
        doc.bounding_box.depth = bbox.get("depth", 0.0)
        # Set target feature version via env override for extractor
        os.environ["FEATURE_VERSION"] = to_version
        new_features = await extractor.extract(doc)
        new_vector = [*new_features.get("geometric", []), *new_features.get("semantic", [])]
        if dry_run:
            feature_migration_total.labels(status="skipped").inc()
        else:
            ok = register_vector(analysis_id, [float(x) for x in new_vector])
            if ok and backend == "faiss":
                try:
                    faiss_store.add(analysis_id, [float(x) for x in new_vector])
                except Exception:
                    pass
            feature_migration_total.labels(status="success" if ok else "error").inc()
        if i % batch_size == 0:
            print(f"Processed {i} vectors...")


def main():
    parser = argparse.ArgumentParser(description="Feature version migration")
    parser.add_argument("--from", dest="from_version", default="v1")
    parser.add_argument("--to", dest="to_version", default="v2")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=50)
    parser.add_argument("--dry-run", dest="dry_run", type=int, default=1)
    args = parser.parse_args()

    # Collect IDs from cache prefix listing not implemented; require external list via env or file
    id_list_file = os.getenv("MIGRATION_ID_LIST_FILE")
    if not id_list_file or not os.path.exists(id_list_file):
        print("MIGRATION_ID_LIST_FILE not set or file missing; aborting")
        return
    with open(id_list_file, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    print(f"Starting migration from {args.from_version} to {args.to_version} (dry_run={args.dry_run}) for {len(ids)} IDs")
    asyncio.run(migrate(ids, args.from_version, args.to_version, bool(args.dry_run), args.batch_size))
    print("Migration finished")


if __name__ == "__main__":
    main()

