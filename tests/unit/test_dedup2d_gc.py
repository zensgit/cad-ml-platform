"""Unit tests for dedup2d uploads GC functionality."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest


class TestRetentionConfig:
    """Tests for retention_seconds configuration."""

    def test_default_retention(self) -> None:
        """Default retention should be 3600 seconds (1 hour)."""
        from src.core.dedup2d_file_storage import Dedup2DFileStorageConfig

        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = Dedup2DFileStorageConfig.from_env()

        assert cfg.retention_seconds == 3600

    def test_custom_retention(self) -> None:
        """Custom retention from env var."""
        from src.core.dedup2d_file_storage import Dedup2DFileStorageConfig

        env = {"DEDUP2D_FILE_STORAGE_RETENTION_SECONDS": "7200"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Dedup2DFileStorageConfig.from_env()

        assert cfg.retention_seconds == 7200

    def test_zero_retention_disables_gc(self) -> None:
        """Zero retention should disable GC."""
        from src.core.dedup2d_file_storage import Dedup2DFileStorageConfig

        env = {"DEDUP2D_FILE_STORAGE_RETENTION_SECONDS": "0"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Dedup2DFileStorageConfig.from_env()

        assert cfg.retention_seconds == 0

    def test_negative_retention_clamped_to_zero(self) -> None:
        """Negative retention should be clamped to 0."""
        from src.core.dedup2d_file_storage import Dedup2DFileStorageConfig

        env = {"DEDUP2D_FILE_STORAGE_RETENTION_SECONDS": "-100"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Dedup2DFileStorageConfig.from_env()

        assert cfg.retention_seconds == 0

    def test_invalid_retention_uses_default(self) -> None:
        """Invalid retention string should use default."""
        from src.core.dedup2d_file_storage import Dedup2DFileStorageConfig

        env = {"DEDUP2D_FILE_STORAGE_RETENTION_SECONDS": "not_a_number"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Dedup2DFileStorageConfig.from_env()

        assert cfg.retention_seconds == 3600  # Default


class TestGCScriptHelpers:
    """Tests for GC script helper functions."""

    def test_format_size_bytes(self) -> None:
        """Format small file sizes."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.dedup2d_uploads_gc import format_size

        assert format_size(0) == "0 B"
        assert format_size(512) == "512 B"
        assert format_size(1023) == "1023 B"

    def test_format_size_kb(self) -> None:
        """Format KB-sized files."""
        from scripts.dedup2d_uploads_gc import format_size

        assert format_size(1024) == "1.0 KB"
        assert format_size(2048) == "2.0 KB"
        assert format_size(1024 * 100) == "100.0 KB"

    def test_format_size_mb(self) -> None:
        """Format MB-sized files."""
        from scripts.dedup2d_uploads_gc import format_size

        assert format_size(1024 * 1024) == "1.0 MB"
        assert format_size(1024 * 1024 * 5) == "5.0 MB"

    def test_format_size_gb(self) -> None:
        """Format GB-sized files."""
        from scripts.dedup2d_uploads_gc import format_size

        assert format_size(1024 * 1024 * 1024) == "1.00 GB"
        assert format_size(1024 * 1024 * 1024 * 2) == "2.00 GB"

    def test_format_age_seconds(self) -> None:
        """Format age in seconds."""
        from scripts.dedup2d_uploads_gc import format_age

        assert format_age(0) == "0s"
        assert format_age(30) == "30s"
        assert format_age(59) == "59s"

    def test_format_age_minutes(self) -> None:
        """Format age in minutes."""
        from scripts.dedup2d_uploads_gc import format_age

        assert format_age(60) == "1m"
        assert format_age(120) == "2m"
        assert format_age(3599) == "59m"

    def test_format_age_hours(self) -> None:
        """Format age in hours."""
        from scripts.dedup2d_uploads_gc import format_age

        assert format_age(3600) == "1.0h"
        assert format_age(7200) == "2.0h"
        assert format_age(86399) == "24.0h"

    def test_format_age_days(self) -> None:
        """Format age in days."""
        from scripts.dedup2d_uploads_gc import format_age

        assert format_age(86400) == "1.0d"
        assert format_age(172800) == "2.0d"


class TestLocalGC:
    """Tests for local filesystem GC."""

    def test_list_local_files_empty_dir(self) -> None:
        """List files in empty directory."""
        from scripts.dedup2d_uploads_gc import list_local_files
        from src.core.dedup2d_file_storage import Dedup2DFileStorageConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "DEDUP2D_FILE_STORAGE": "local",
                "DEDUP2D_FILE_STORAGE_DIR": tmpdir,
            }
            with mock.patch.dict(os.environ, env, clear=True):
                cfg = Dedup2DFileStorageConfig.from_env()

            files = list(list_local_files(cfg))
            assert len(files) == 0

    def test_list_local_files_with_files(self) -> None:
        """List files in directory with content."""
        from scripts.dedup2d_uploads_gc import list_local_files
        from src.core.dedup2d_file_storage import Dedup2DFileStorageConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some test files
            (Path(tmpdir) / "file1.txt").write_text("hello")
            (Path(tmpdir) / "subdir").mkdir()
            (Path(tmpdir) / "subdir" / "file2.txt").write_text("world")

            env = {
                "DEDUP2D_FILE_STORAGE": "local",
                "DEDUP2D_FILE_STORAGE_DIR": tmpdir,
            }
            with mock.patch.dict(os.environ, env, clear=True):
                cfg = Dedup2DFileStorageConfig.from_env()

            files = list(list_local_files(cfg))
            assert len(files) == 2
            keys = {f.key for f in files}
            assert "file1.txt" in keys
            assert "subdir/file2.txt" in keys or "subdir\\file2.txt" in keys

    def test_list_local_files_nonexistent_dir(self) -> None:
        """List files when directory doesn't exist."""
        from scripts.dedup2d_uploads_gc import list_local_files
        from src.core.dedup2d_file_storage import Dedup2DFileStorageConfig

        env = {
            "DEDUP2D_FILE_STORAGE": "local",
            "DEDUP2D_FILE_STORAGE_DIR": "/nonexistent/path/to/dir",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Dedup2DFileStorageConfig.from_env()

        files = list(list_local_files(cfg))
        assert len(files) == 0

    def test_delete_local_file(self) -> None:
        """Delete a local file."""
        from scripts.dedup2d_uploads_gc import delete_local_file
        from src.core.dedup2d_file_storage import Dedup2DFileStorageConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("content")

            env = {
                "DEDUP2D_FILE_STORAGE": "local",
                "DEDUP2D_FILE_STORAGE_DIR": tmpdir,
            }
            with mock.patch.dict(os.environ, env, clear=True):
                cfg = Dedup2DFileStorageConfig.from_env()

            assert test_file.exists()
            result = delete_local_file(cfg, "test.txt")
            assert result is True
            assert not test_file.exists()

    def test_delete_local_file_removes_empty_parent(self) -> None:
        """Deleting file should remove empty parent directories."""
        from scripts.dedup2d_uploads_gc import delete_local_file
        from src.core.dedup2d_file_storage import Dedup2DFileStorageConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "job123" / "uuid"
            subdir.mkdir(parents=True)
            test_file = subdir / "test.txt"
            test_file.write_text("content")

            env = {
                "DEDUP2D_FILE_STORAGE": "local",
                "DEDUP2D_FILE_STORAGE_DIR": tmpdir,
            }
            with mock.patch.dict(os.environ, env, clear=True):
                cfg = Dedup2DFileStorageConfig.from_env()

            delete_local_file(cfg, "job123/uuid/test.txt")

            # File and empty parents should be removed
            assert not test_file.exists()
            assert not subdir.exists()

    def test_delete_local_file_path_traversal_blocked(self) -> None:
        """Path traversal attempts should be blocked."""
        from scripts.dedup2d_uploads_gc import delete_local_file
        from src.core.dedup2d_file_storage import Dedup2DFileStorageConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "DEDUP2D_FILE_STORAGE": "local",
                "DEDUP2D_FILE_STORAGE_DIR": tmpdir,
            }
            with mock.patch.dict(os.environ, env, clear=True):
                cfg = Dedup2DFileStorageConfig.from_env()

            # Attempt path traversal
            result = delete_local_file(cfg, "../../../etc/passwd")
            assert result is False


class TestGCDryRun:
    """Tests for GC dry run functionality."""

    def test_gc_dry_run_does_not_delete(self) -> None:
        """Dry run should not delete any files."""
        from scripts.dedup2d_uploads_gc import run_gc
        from src.core.dedup2d_file_storage import Dedup2DFileStorageConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create old file
            test_file = Path(tmpdir) / "old_file.txt"
            test_file.write_text("content")
            # Make file appear old
            old_time = time.time() - 7200  # 2 hours ago
            os.utime(test_file, (old_time, old_time))

            env = {
                "DEDUP2D_FILE_STORAGE": "local",
                "DEDUP2D_FILE_STORAGE_DIR": tmpdir,
            }
            with mock.patch.dict(os.environ, env, clear=True):
                cfg = Dedup2DFileStorageConfig.from_env()

            # Run dry run
            total, deleted, deleted_bytes = run_gc(
                config=cfg,
                retention_seconds=3600,  # 1 hour
                execute=False,  # Dry run
                verbose=False,
            )

            # File should still exist
            assert test_file.exists()
            assert total == 1
            assert deleted == 1  # Would be deleted
            assert deleted_bytes > 0

    def test_gc_execute_deletes_old_files(self) -> None:
        """Execute mode should delete old files."""
        from scripts.dedup2d_uploads_gc import run_gc
        from src.core.dedup2d_file_storage import Dedup2DFileStorageConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create old file
            old_file = Path(tmpdir) / "old_file.txt"
            old_file.write_text("content")
            old_time = time.time() - 7200  # 2 hours ago
            os.utime(old_file, (old_time, old_time))

            # Create new file
            new_file = Path(tmpdir) / "new_file.txt"
            new_file.write_text("content")

            env = {
                "DEDUP2D_FILE_STORAGE": "local",
                "DEDUP2D_FILE_STORAGE_DIR": tmpdir,
            }
            with mock.patch.dict(os.environ, env, clear=True):
                cfg = Dedup2DFileStorageConfig.from_env()

            # Run execute
            total, deleted, deleted_bytes = run_gc(
                config=cfg,
                retention_seconds=3600,  # 1 hour
                execute=True,
                verbose=False,
            )

            # Old file should be deleted, new file should remain
            assert not old_file.exists()
            assert new_file.exists()
            assert total == 2
            assert deleted == 1
