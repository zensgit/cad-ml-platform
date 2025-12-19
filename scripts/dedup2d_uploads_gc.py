#!/usr/bin/env python3
"""
Garbage collection script for dedup2d uploaded files.

Deletes files older than DEDUP2D_FILE_STORAGE_RETENTION_SECONDS.
Supports both local filesystem and S3/MinIO backends.

Usage:
    # Dry run (default) - just show what would be deleted
    python scripts/dedup2d_uploads_gc.py

    # Actually delete files
    python scripts/dedup2d_uploads_gc.py --execute

    # Custom retention (override env var)
    python scripts/dedup2d_uploads_gc.py --retention-seconds 7200 --execute

    # Verbose output
    python scripts/dedup2d_uploads_gc.py --verbose --execute

Environment Variables:
    DEDUP2D_FILE_STORAGE          Backend: "local" or "s3"
    DEDUP2D_FILE_STORAGE_DIR      Local storage directory
    DEDUP2D_FILE_STORAGE_RETENTION_SECONDS  Retention period (default: 3600)
    DEDUP2D_S3_BUCKET             S3 bucket name
    DEDUP2D_S3_PREFIX             S3 key prefix
    DEDUP2D_S3_ENDPOINT           S3 endpoint (for MinIO)
    DEDUP2D_S3_REGION             S3 region

Kubernetes CronJob Example:
    apiVersion: batch/v1
    kind: CronJob
    metadata:
      name: dedup2d-uploads-gc
    spec:
      schedule: "0 * * * *"  # Every hour
      jobTemplate:
        spec:
          template:
            spec:
              containers:
              - name: gc
                image: cad-ml-platform:latest
                command: ["python", "scripts/dedup2d_uploads_gc.py", "--execute"]
                envFrom:
                - configMapRef:
                    name: cad-ml-config
              restartPolicy: OnFailure
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.core.dedup2d_file_storage import Dedup2DFileStorageConfig

logger = logging.getLogger(__name__)


class FileInfo(NamedTuple):
    """Information about a file to potentially delete."""
    key: str  # path (local) or s3 key
    mtime: float  # modification time (unix timestamp)
    size: int  # file size in bytes


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def list_local_files(config: Dedup2DFileStorageConfig) -> Iterator[FileInfo]:
    """List all files in local storage with their metadata."""
    base_dir = Path(config.local_dir)
    if not base_dir.exists():
        logger.info(f"Local storage directory does not exist: {base_dir}")
        return

    for file_path in base_dir.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            stat = file_path.stat()
            relative_path = str(file_path.relative_to(base_dir))
            yield FileInfo(
                key=relative_path,
                mtime=stat.st_mtime,
                size=stat.st_size,
            )
        except OSError as e:
            logger.warning(f"Could not stat {file_path}: {e}")


def list_s3_files(config: Dedup2DFileStorageConfig) -> Iterator[FileInfo]:
    """List all objects in S3 bucket with their metadata."""
    try:
        import boto3
    except ImportError:
        logger.error("boto3 is required for S3 backend. Install with: pip install boto3")
        return

    if not config.s3_bucket:
        logger.error("DEDUP2D_S3_BUCKET is required for S3 backend")
        return

    client = boto3.client(
        "s3",
        endpoint_url=config.s3_endpoint,
        region_name=config.s3_region,
    )

    prefix = config.s3_prefix.strip("/")
    if prefix:
        prefix = prefix + "/"

    paginator = client.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=config.s3_bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj.get("Key", "")
                last_modified = obj.get("LastModified")
                size = obj.get("Size", 0)

                if last_modified is None:
                    continue

                # Convert datetime to unix timestamp
                mtime = last_modified.timestamp()

                yield FileInfo(key=key, mtime=mtime, size=size)
    except Exception as e:
        logger.error(f"Failed to list S3 objects: {e}")


def delete_local_file(config: Dedup2DFileStorageConfig, key: str) -> bool:
    """Delete a local file. Returns True on success."""
    base_dir = Path(config.local_dir)
    file_path = base_dir / key

    # Security check - ensure path is within base_dir
    try:
        file_path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        logger.warning(f"Refusing to delete file outside base directory: {key}")
        return False

    try:
        file_path.unlink()
        # Try to remove empty parent directories
        parent = file_path.parent
        while parent != base_dir:
            try:
                parent.rmdir()
                parent = parent.parent
            except OSError:
                break
        return True
    except FileNotFoundError:
        return True  # Already deleted
    except OSError as e:
        logger.error(f"Failed to delete {key}: {e}")
        return False


def delete_s3_file(config: Dedup2DFileStorageConfig, key: str) -> bool:
    """Delete an S3 object. Returns True on success."""
    try:
        import boto3
    except ImportError:
        return False

    client = boto3.client(
        "s3",
        endpoint_url=config.s3_endpoint,
        region_name=config.s3_region,
    )

    try:
        client.delete_object(Bucket=config.s3_bucket, Key=key)
        return True
    except Exception as e:
        logger.error(f"Failed to delete s3://{config.s3_bucket}/{key}: {e}")
        return False


def format_size(size: int) -> str:
    """Format file size for display."""
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.2f} GB"


def format_age(age_seconds: float) -> str:
    """Format file age for display."""
    if age_seconds < 60:
        return f"{int(age_seconds)}s"
    elif age_seconds < 3600:
        return f"{int(age_seconds / 60)}m"
    elif age_seconds < 86400:
        return f"{age_seconds / 3600:.1f}h"
    else:
        return f"{age_seconds / 86400:.1f}d"


def run_gc(
    config: Dedup2DFileStorageConfig,
    retention_seconds: int,
    execute: bool,
    verbose: bool,
) -> tuple[int, int, int]:
    """
    Run garbage collection.

    Returns:
        Tuple of (total_files, deleted_files, deleted_bytes)
    """
    now = time.time()
    cutoff_time = now - retention_seconds
    cutoff_dt = datetime.fromtimestamp(cutoff_time, tz=timezone.utc)

    backend = config.backend.lower()
    logger.info(f"Starting GC for backend={backend}")
    logger.info(f"Retention: {retention_seconds}s ({format_age(float(retention_seconds))})")
    logger.info(f"Cutoff time: {cutoff_dt.isoformat()}")
    logger.info(f"Mode: {'EXECUTE' if execute else 'DRY RUN'}")

    # List files based on backend
    if backend == "local":
        files = list_local_files(config)
        delete_fn = delete_local_file
    elif backend == "s3":
        files = list_s3_files(config)
        delete_fn = delete_s3_file
    else:
        logger.error(f"Unknown backend: {backend}")
        return 0, 0, 0

    total_files = 0
    deleted_files = 0
    deleted_bytes = 0

    for file_info in files:
        total_files += 1
        age = now - file_info.mtime

        if file_info.mtime < cutoff_time:
            # File is older than retention period
            if verbose:
                logger.debug(
                    f"{'DELETE' if execute else 'WOULD DELETE'}: "
                    f"{file_info.key} (age={format_age(age)}, size={format_size(file_info.size)})"
                )

            if execute:
                if delete_fn(config, file_info.key):
                    deleted_files += 1
                    deleted_bytes += file_info.size
            else:
                deleted_files += 1
                deleted_bytes += file_info.size
        elif verbose:
            logger.debug(
                f"KEEP: {file_info.key} (age={format_age(age)}, size={format_size(file_info.size)})"
            )

    return total_files, deleted_files, deleted_bytes


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Garbage collection for dedup2d uploaded files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files (default: dry run)",
    )
    parser.add_argument(
        "--retention-seconds",
        type=int,
        default=None,
        help="Override DEDUP2D_FILE_STORAGE_RETENTION_SECONDS",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output (each file)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Load config
    config = Dedup2DFileStorageConfig.from_env()
    retention = args.retention_seconds if args.retention_seconds is not None else config.retention_seconds

    if retention <= 0:
        logger.warning("Retention is 0 or negative - GC disabled")
        return 0

    # Run GC
    total, deleted, deleted_bytes = run_gc(
        config=config,
        retention_seconds=retention,
        execute=args.execute,
        verbose=args.verbose,
    )

    # Summary
    logger.info("=" * 50)
    logger.info(f"Total files scanned: {total}")
    logger.info(f"Files {'deleted' if args.execute else 'to delete'}: {deleted}")
    logger.info(f"Space {'freed' if args.execute else 'to free'}: {format_size(deleted_bytes)}")

    if not args.execute and deleted > 0:
        logger.info("")
        logger.info("This was a DRY RUN. Use --execute to actually delete files.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
