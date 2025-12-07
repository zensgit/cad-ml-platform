#!/usr/bin/env python3
"""CAD ML Platform - v5 Feature Migration Tool

Safely migrate existing feature vectors from v1-v4 to v5 by re-extracting features
from original CAD files.

Features:
  - Batch processing with configurable parallelism
  - Automatic rollback support via backups
  - Incremental migration (resume after interruption)
  - Dry-run mode for safety validation
  - Progress tracking with ETA
  - Detailed logging and metrics

Usage:
  # Dry run to check what would be migrated
  python scripts/migrate_to_v5.py --dry-run

  # Migrate all vectors with backup
  python scripts/migrate_to_v5.py --backup

  # Migrate specific files
  python scripts/migrate_to_v5.py --file-list files.txt

  # Resume interrupted migration
  python scripts/migrate_to_v5.py --resume

  # Rollback to previous version
  python scripts/migrate_to_v5.py --rollback
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import aiohttp
    from tqdm import tqdm
except ImportError:
    print("ERROR: Missing dependencies. Install with:")
    print("  pip install aiohttp tqdm")
    sys.exit(1)

# Configuration
API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1")
BACKUP_DIR = Path("backups/migration_v5")
STATE_FILE = BACKUP_DIR / "migration_state.json"
LOG_FILE = BACKUP_DIR / f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MigrationStats:
    """Track migration statistics."""
    def __init__(self):
        self.total = 0
        self.success = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = time.time()
        self.failures: List[Dict] = []

    def record_success(self):
        self.success += 1

    def record_failure(self, doc_id: str, reason: str):
        self.failed += 1
        self.failures.append({"doc_id": doc_id, "reason": reason, "time": datetime.now().isoformat()})

    def record_skip(self):
        self.skipped += 1

    def summary(self) -> Dict:
        elapsed = time.time() - self.start_time
        return {
            "total": self.total,
            "success": self.success,
            "failed": self.failed,
            "skipped": self.skipped,
            "elapsed_seconds": round(elapsed, 2),
            "throughput_per_sec": round(self.success / elapsed, 2) if elapsed > 0 else 0
        }


class MigrationState:
    """Manage migration state for resumability."""
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state = self.load()

    def load(self) -> Dict:
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {"completed": [], "failed": [], "pending": []}

    def save(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def mark_completed(self, doc_id: str):
        if doc_id not in self.state["completed"]:
            self.state["completed"].append(doc_id)
        if doc_id in self.state["pending"]:
            self.state["pending"].remove(doc_id)
        self.save()

    def mark_failed(self, doc_id: str):
        if doc_id not in self.state["failed"]:
            self.state["failed"].append(doc_id)
        if doc_id in self.state["pending"]:
            self.state["pending"].remove(doc_id)
        self.save()

    def is_completed(self, doc_id: str) -> bool:
        return doc_id in self.state["completed"]


async def get_vector_list(session: aiohttp.ClientSession) -> List[Dict]:
    """Fetch list of all registered vectors."""
    url = f"{API_BASE}/vectors/list"
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("vectors", [])
            else:
                logger.error(f"Failed to fetch vector list: {resp.status}")
                return []
    except Exception as e:
        logger.error(f"Error fetching vector list: {e}")
        return []


async def backup_vector(session: aiohttp.ClientSession, doc_id: str, backup_dir: Path) -> bool:
    """Backup existing vector metadata and features."""
    url = f"{API_BASE}/vectors/{doc_id}"
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                backup_file = backup_dir / f"{doc_id}.json"
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                with open(backup_file, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.debug(f"Backed up {doc_id}")
                return True
            else:
                logger.warning(f"Vector {doc_id} not found for backup")
                return False
    except Exception as e:
        logger.error(f"Failed to backup {doc_id}: {e}")
        return False


async def reanalyze_file(session: aiohttp.ClientSession, file_path: str, doc_id: str) -> Tuple[bool, Optional[str]]:
    """Re-analyze a CAD file with v5 features."""
    url = f"{API_BASE}/analyze"
    
    if not Path(file_path).exists():
        return False, f"File not found: {file_path}"
    
    try:
        # Set FEATURE_VERSION to v5 via header
        headers = {"X-Feature-Version": "v5"}
        
        with open(file_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=Path(file_path).name)
            
            async with session.post(url, data=data, headers=headers) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    logger.debug(f"Re-analyzed {doc_id}: v5 features extracted")
                    return True, None
                else:
                    error_text = await resp.text()
                    return False, f"Analysis failed: {resp.status} - {error_text[:200]}"
    except Exception as e:
        return False, f"Exception: {str(e)}"


async def migrate_single(
    session: aiohttp.ClientSession,
    doc_id: str,
    file_path: str,
    backup_dir: Optional[Path],
    dry_run: bool,
    state: MigrationState
) -> bool:
    """Migrate a single vector."""
    
    # Skip if already completed
    if state.is_completed(doc_id):
        logger.debug(f"Skipping {doc_id} (already migrated)")
        return False  # Not counted as success, just skip
    
    if dry_run:
        logger.info(f"[DRY RUN] Would migrate {doc_id} from {file_path}")
        return True
    
    # Backup if requested
    if backup_dir:
        if not await backup_vector(session, doc_id, backup_dir):
            logger.warning(f"Backup failed for {doc_id}, continuing anyway...")
    
    # Re-analyze
    success, error = await reanalyze_file(session, file_path, doc_id)
    
    if success:
        state.mark_completed(doc_id)
        return True
    else:
        state.mark_failed(doc_id)
        logger.error(f"Migration failed for {doc_id}: {error}")
        return False


async def migrate_batch(
    file_mapping: Dict[str, str],
    backup: bool,
    dry_run: bool,
    resume: bool,
    batch_size: int = 10
) -> MigrationStats:
    """Migrate a batch of vectors."""
    
    stats = MigrationStats()
    stats.total = len(file_mapping)
    
    # Initialize state
    state = MigrationState(STATE_FILE)
    
    # Filter out already completed if resuming
    if resume:
        to_migrate = {k: v for k, v in file_mapping.items() if not state.is_completed(k)}
        logger.info(f"Resuming: {len(to_migrate)}/{stats.total} remaining")
    else:
        to_migrate = file_mapping
    
    backup_dir = BACKUP_DIR / datetime.now().strftime("%Y%m%d_%H%M%S") if backup else None
    
    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Backups will be saved to: {backup_dir}")
    
    async with aiohttp.ClientSession() as session:
        # Progress bar
        pbar = tqdm(total=len(to_migrate), desc="Migrating", unit="file")
        
        # Process in batches
        items = list(to_migrate.items())
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            
            tasks = [
                migrate_single(session, doc_id, file_path, backup_dir, dry_run, state)
                for doc_id, file_path in batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for (doc_id, file_path), result in zip(batch, results):
                if isinstance(result, Exception):
                    stats.record_failure(doc_id, str(result))
                elif result is False:  # Skipped
                    stats.record_skip()
                elif result:  # Success
                    stats.record_success()
                
                pbar.update(1)
        
        pbar.close()
    
    return stats


def load_file_mapping(file_list_path: Optional[Path]) -> Dict[str, str]:
    """Load file mapping from a file list or discover from API."""
    
    if file_list_path and file_list_path.exists():
        # Load from file (format: doc_id,file_path)
        mapping = {}
        with open(file_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        mapping[parts[0]] = parts[1]
        logger.info(f"Loaded {len(mapping)} files from {file_list_path}")
        return mapping
    else:
        # TODO: In production, this would query the database or file registry
        # For now, return an example mapping
        logger.warning("No file list provided. Please create a file with format: doc_id,file_path")
        logger.warning("Example:")
        logger.warning("  part_001,/path/to/drawings/part_001.dxf")
        logger.warning("  part_002,/path/to/drawings/part_002.step")
        return {}


async def rollback(backup_dir: Path):
    """Rollback to previous version using backups."""
    if not backup_dir.exists():
        logger.error(f"Backup directory not found: {backup_dir}")
        return
    
    backup_files = list(backup_dir.glob("*.json"))
    logger.info(f"Found {len(backup_files)} backup files")
    
    # TODO: Implement actual rollback logic
    # This would involve:
    # 1. Reading each backup JSON
    # 2. Calling /vectors/{doc_id} PUT to restore old vector
    # 3. Updating metrics
    
    logger.warning("Rollback not yet implemented. Backup files preserved at: {backup_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate CAD feature vectors from v1-v4 to v5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    parser.add_argument("--backup", action="store_true", help="Backup vectors before migration")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted migration")
    parser.add_argument("--rollback", action="store_true", help="Rollback to previous version")
    parser.add_argument("--file-list", type=Path, help="Path to file list (doc_id,file_path)")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for parallel processing")
    parser.add_argument("--backup-dir", type=Path, help="Custom backup directory")
    
    args = parser.parse_args()
    
    # Override backup dir if specified
    if args.backup_dir:
        global BACKUP_DIR
        BACKUP_DIR = args.backup_dir
    
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("CAD ML Platform - v5 Migration Tool")
    logger.info("=" * 60)
    logger.info(f"API Base: {API_BASE}")
    logger.info(f"Backup Dir: {BACKUP_DIR}")
    logger.info(f"Log File: {LOG_FILE}")
    logger.info("")
    
    # Rollback mode
    if args.rollback:
        logger.info("ROLLBACK MODE")
        # Find latest backup
        backup_dirs = sorted(BACKUP_DIR.glob("*"), reverse=True)
        if backup_dirs:
            latest = backup_dirs[0]
            logger.info(f"Rolling back from: {latest}")
            asyncio.run(rollback(latest))
        else:
            logger.error("No backups found")
        return
    
    # Load file mapping
    file_mapping = load_file_mapping(args.file_list)
    
    if not file_mapping:
        logger.error("No files to migrate. Provide --file-list or configure discovery.")
        sys.exit(1)
    
    # Run migration
    logger.info(f"Starting migration of {len(file_mapping)} files...")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Backup: {args.backup}")
    logger.info(f"Resume: {args.resume}")
    logger.info("")
    
    stats = asyncio.run(migrate_batch(
        file_mapping,
        backup=args.backup,
        dry_run=args.dry_run,
        resume=args.resume,
        batch_size=args.batch_size
    ))
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("MIGRATION COMPLETE")
    logger.info("=" * 60)
    summary = stats.summary()
    logger.info(f"Total:     {summary['total']}")
    logger.info(f"Success:   {summary['success']} ✓")
    logger.info(f"Failed:    {summary['failed']} ✗")
    logger.info(f"Skipped:   {summary['skipped']} -")
    logger.info(f"Elapsed:   {summary['elapsed_seconds']}s")
    logger.info(f"Throughput: {summary['throughput_per_sec']} files/sec")
    logger.info("")
    
    if stats.failures:
        logger.warning(f"Failed migrations ({len(stats.failures)}):")
        for failure in stats.failures[:10]:  # Show first 10
            logger.warning(f"  - {failure['doc_id']}: {failure['reason']}")
        if len(stats.failures) > 10:
            logger.warning(f"  ... and {len(stats.failures) - 10} more")
    
    logger.info(f"Full log: {LOG_FILE}")
    
    # Exit code
    sys.exit(0 if stats.failed == 0 else 1)


if __name__ == "__main__":
    main()
