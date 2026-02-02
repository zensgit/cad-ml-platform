"""
DWG file manager for batch operations and caching.

Provides high-level management of DWG files including
conversion, caching, and batch processing.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import threading
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

from src.core.cad.dwg.converter import (
    DWGConverter,
    ConverterConfig,
    ConversionResult,
    ConversionStatus,
    DXFVersion,
)
from src.core.cad.dwg.parser import DWGParser, DWGHeader, DWGVersion

logger = logging.getLogger(__name__)


@dataclass
class DWGFile:
    """Representation of a DWG file with metadata."""
    path: Path
    header: Optional[DWGHeader] = None
    dxf_path: Optional[Path] = None
    conversion_result: Optional[ConversionResult] = None
    last_accessed: datetime = field(default_factory=datetime.now)
    file_hash: Optional[str] = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def exists(self) -> bool:
        return self.path.exists()

    @property
    def is_converted(self) -> bool:
        return self.dxf_path is not None and self.dxf_path.exists()

    @property
    def version(self) -> DWGVersion:
        return self.header.version if self.header else DWGVersion.UNKNOWN

    @property
    def size(self) -> int:
        return self.path.stat().st_size if self.exists else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "name": self.name,
            "version": self.version.value,
            "is_converted": self.is_converted,
            "dxf_path": str(self.dxf_path) if self.dxf_path else None,
            "size": self.size,
            "last_accessed": self.last_accessed.isoformat(),
            "file_hash": self.file_hash,
        }


@dataclass
class ManagerConfig:
    """Configuration for DWG manager."""
    cache_dir: Optional[str] = None
    cache_ttl_hours: int = 24
    max_cache_size_mb: int = 1000
    auto_convert: bool = True
    converter_config: Optional[ConverterConfig] = None
    parallel_conversions: int = 4


class DWGManager:
    """
    High-level manager for DWG files.

    Provides:
    - File discovery and tracking
    - Conversion caching
    - Batch operations
    - Memory management
    """

    def __init__(self, config: Optional[ManagerConfig] = None):
        """
        Initialize DWG manager.

        Args:
            config: Manager configuration
        """
        self._config = config or ManagerConfig()
        self._converter = DWGConverter(self._config.converter_config)
        self._files: Dict[str, DWGFile] = {}
        self._cache_dir: Optional[Path] = None
        self._lock = threading.RLock()

        self._setup_cache()

    def _setup_cache(self) -> None:
        """Setup cache directory."""
        if self._config.cache_dir:
            self._cache_dir = Path(self._config.cache_dir)
        else:
            # Use system temp directory
            self._cache_dir = Path(tempfile.gettempdir()) / "dwg_cache"

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DWG cache directory: {self._cache_dir}")

    @property
    def converter(self) -> DWGConverter:
        """Get the underlying converter."""
        return self._converter

    @property
    def cache_dir(self) -> Optional[Path]:
        """Get cache directory."""
        return self._cache_dir

    @property
    def is_converter_available(self) -> bool:
        """Check if converter is available."""
        return self._converter.is_available

    def register(self, file_path: Union[str, Path]) -> DWGFile:
        """
        Register a DWG file with the manager.

        Args:
            file_path: Path to DWG file

        Returns:
            DWGFile object
        """
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"DWG file not found: {path}")

        if not path.suffix.lower() == ".dwg":
            raise ValueError(f"Not a DWG file: {path}")

        key = str(path)

        with self._lock:
            if key in self._files:
                dwg_file = self._files[key]
                dwg_file.last_accessed = datetime.now()
                return dwg_file

            # Parse header
            try:
                parser = DWGParser(path)
                header = parser.header
            except Exception as e:
                logger.warning(f"Failed to parse DWG header: {e}")
                header = None

            # Calculate file hash
            file_hash = self._calculate_hash(path)

            # Check for cached conversion
            dxf_path = self._get_cached_dxf(file_hash)

            dwg_file = DWGFile(
                path=path,
                header=header,
                dxf_path=dxf_path,
                file_hash=file_hash,
            )

            self._files[key] = dwg_file

            # Auto-convert if enabled
            if self._config.auto_convert and not dwg_file.is_converted:
                self.convert(dwg_file)

            return dwg_file

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate file hash for caching."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_cached_dxf(self, file_hash: str) -> Optional[Path]:
        """Get cached DXF file if exists."""
        if not self._cache_dir or not file_hash:
            return None

        cached_path = self._cache_dir / f"{file_hash}.dxf"
        if cached_path.exists():
            # Check TTL
            mtime = datetime.fromtimestamp(cached_path.stat().st_mtime)
            if datetime.now() - mtime < timedelta(hours=self._config.cache_ttl_hours):
                return cached_path
            else:
                # Expired, remove
                cached_path.unlink(missing_ok=True)

        return None

    def convert(
        self,
        dwg_file: Union[DWGFile, str, Path],
        output_path: Optional[Union[str, Path]] = None,
        version: Optional[DXFVersion] = None,
        use_cache: bool = True,
    ) -> ConversionResult:
        """
        Convert a DWG file to DXF.

        Args:
            dwg_file: DWGFile object or path
            output_path: Output path (uses cache if None)
            version: DXF version
            use_cache: Whether to use/store in cache

        Returns:
            ConversionResult
        """
        # Resolve DWGFile
        if isinstance(dwg_file, (str, Path)):
            dwg_file = self.register(dwg_file)

        # Check cache
        if use_cache and dwg_file.is_converted and output_path is None:
            return ConversionResult(
                input_path=str(dwg_file.path),
                output_path=str(dwg_file.dxf_path),
                status=ConversionStatus.SUCCESS,
            )

        # Determine output path
        if output_path is None and self._cache_dir and dwg_file.file_hash:
            output_path = self._cache_dir / f"{dwg_file.file_hash}.dxf"
        elif output_path is None:
            output_path = dwg_file.path.with_suffix(".dxf")

        # Convert
        result = self._converter.convert(dwg_file.path, output_path, version)

        # Update file record
        if result.success:
            dwg_file.dxf_path = Path(result.output_path) if result.output_path else None
            dwg_file.conversion_result = result

        return result

    def get_dxf(self, dwg_file: Union[DWGFile, str, Path]) -> Optional[Path]:
        """
        Get DXF file for a DWG, converting if necessary.

        Args:
            dwg_file: DWGFile object or path

        Returns:
            Path to DXF file or None
        """
        if isinstance(dwg_file, (str, Path)):
            dwg_file = self.register(dwg_file)

        if not dwg_file.is_converted:
            result = self.convert(dwg_file)
            if not result.success:
                return None

        return dwg_file.dxf_path

    def discover(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        pattern: str = "*.dwg",
    ) -> List[DWGFile]:
        """
        Discover DWG files in a directory.

        Args:
            directory: Directory to search
            recursive: Search subdirectories
            pattern: Glob pattern

        Returns:
            List of DWGFile objects
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))

        results = []
        for file_path in files:
            try:
                dwg_file = self.register(file_path)
                results.append(dwg_file)
            except Exception as e:
                logger.warning(f"Failed to register {file_path}: {e}")

        logger.info(f"Discovered {len(results)} DWG files in {directory}")
        return results

    def batch_convert(
        self,
        files: Optional[List[Union[DWGFile, str, Path]]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        version: Optional[DXFVersion] = None,
        max_workers: Optional[int] = None,
    ) -> List[ConversionResult]:
        """
        Batch convert multiple DWG files.

        Args:
            files: List of files to convert (uses all registered if None)
            output_dir: Output directory
            version: DXF version
            max_workers: Maximum parallel conversions

        Returns:
            List of ConversionResult
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if files is None:
            files = list(self._files.values())

        max_workers = max_workers or self._config.parallel_conversions
        results = []

        def convert_one(f: Union[DWGFile, str, Path]) -> ConversionResult:
            if isinstance(f, (str, Path)):
                f = self.register(f)

            out_path = None
            if output_dir:
                out_path = Path(output_dir) / f"{f.stem}.dxf"

            return self.convert(f, out_path, version, use_cache=output_dir is None)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(convert_one, f): f for f in files}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    f = futures[future]
                    path = f.path if isinstance(f, DWGFile) else f
                    results.append(ConversionResult(
                        input_path=str(path),
                        status=ConversionStatus.FAILED,
                        error_message=str(e),
                    ))

        # Summary
        success = sum(1 for r in results if r.success)
        logger.info(f"Batch conversion: {success}/{len(results)} successful")

        return results

    def get_file(self, path: Union[str, Path]) -> Optional[DWGFile]:
        """Get registered file by path."""
        key = str(Path(path).resolve())
        return self._files.get(key)

    def list_files(self) -> List[DWGFile]:
        """List all registered files."""
        return list(self._files.values())

    def iter_files(self) -> Iterator[DWGFile]:
        """Iterate over registered files."""
        return iter(self._files.values())

    def unregister(self, path: Union[str, Path]) -> bool:
        """
        Unregister a file.

        Args:
            path: File path

        Returns:
            True if file was registered
        """
        key = str(Path(path).resolve())
        with self._lock:
            if key in self._files:
                del self._files[key]
                return True
        return False

    def clear(self) -> None:
        """Clear all registered files."""
        with self._lock:
            self._files.clear()

    def clear_cache(self, older_than_hours: Optional[int] = None) -> int:
        """
        Clear conversion cache.

        Args:
            older_than_hours: Only clear files older than this (None for all)

        Returns:
            Number of files removed
        """
        if not self._cache_dir:
            return 0

        count = 0
        cutoff = None
        if older_than_hours:
            cutoff = datetime.now() - timedelta(hours=older_than_hours)

        for file_path in self._cache_dir.glob("*.dxf"):
            try:
                if cutoff:
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mtime >= cutoff:
                        continue

                file_path.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete cache file {file_path}: {e}")

        logger.info(f"Cleared {count} cached files")
        return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._cache_dir:
            return {"enabled": False}

        files = list(self._cache_dir.glob("*.dxf"))
        total_size = sum(f.stat().st_size for f in files)

        return {
            "enabled": True,
            "directory": str(self._cache_dir),
            "file_count": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "max_size_mb": self._config.max_cache_size_mb,
            "ttl_hours": self._config.cache_ttl_hours,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        files = list(self._files.values())
        converted = sum(1 for f in files if f.is_converted)

        return {
            "registered_files": len(files),
            "converted_files": converted,
            "converter_available": self.is_converter_available,
            "cache": self.get_cache_stats(),
        }

    def get_info(self) -> Dict[str, Any]:
        """Get detailed manager information."""
        return {
            "stats": self.get_stats(),
            "converter": self._converter.get_info(),
            "config": {
                "cache_dir": str(self._cache_dir) if self._cache_dir else None,
                "cache_ttl_hours": self._config.cache_ttl_hours,
                "max_cache_size_mb": self._config.max_cache_size_mb,
                "auto_convert": self._config.auto_convert,
                "parallel_conversions": self._config.parallel_conversions,
            },
        }


# Global manager instance
_global_manager: Optional[DWGManager] = None
_manager_lock = threading.Lock()


def get_dwg_manager(config: Optional[ManagerConfig] = None) -> DWGManager:
    """
    Get global DWG manager instance.

    Args:
        config: Configuration (only used for first call)

    Returns:
        DWGManager instance
    """
    global _global_manager

    with _manager_lock:
        if _global_manager is None:
            _global_manager = DWGManager(config)
        return _global_manager


def reset_manager() -> None:
    """Reset global manager (mainly for testing)."""
    global _global_manager
    with _manager_lock:
        if _global_manager:
            _global_manager.clear()
        _global_manager = None
