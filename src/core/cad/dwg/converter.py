"""
DWG to DXF converter using ODA File Converter.

Provides integration with ODA File Converter for DWG/DXF conversion.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

logger = logging.getLogger(__name__)


class DXFVersion(str, Enum):
    """Supported DXF output versions."""
    R12 = "ACAD12"
    R13 = "ACAD13"
    R14 = "ACAD14"
    R2000 = "ACAD2000"
    R2004 = "ACAD2004"
    R2007 = "ACAD2007"
    R2010 = "ACAD2010"
    R2013 = "ACAD2013"
    R2018 = "ACAD2018"


class ConversionStatus(str, Enum):
    """Conversion status."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    CONVERTER_NOT_FOUND = "converter_not_found"


@dataclass
class ConverterConfig:
    """Configuration for DWG converter."""
    oda_path: Optional[str] = None  # Path to ODA File Converter
    output_version: DXFVersion = DXFVersion.R2018
    output_format: str = "DXF"  # DXF or DWG
    audit: bool = True  # Audit and fix errors
    recursive: bool = False  # Process subdirectories
    timeout: int = 60  # Conversion timeout in seconds
    temp_dir: Optional[str] = None  # Temporary directory for conversions


@dataclass
class ConversionResult:
    """Result of a DWG conversion."""
    input_path: str
    output_path: Optional[str] = None
    status: ConversionStatus = ConversionStatus.SUCCESS
    error_message: Optional[str] = None
    conversion_time: float = 0.0
    file_size_before: int = 0
    file_size_after: int = 0

    @property
    def success(self) -> bool:
        return self.status == ConversionStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "status": self.status.value,
            "error_message": self.error_message,
            "conversion_time": round(self.conversion_time, 3),
            "file_size_before": self.file_size_before,
            "file_size_after": self.file_size_after,
        }


class DWGConverter:
    """
    DWG to DXF converter.

    Uses ODA File Converter for reliable conversion.
    Falls back to basic parsing for simple operations.
    """

    # Default ODA paths by platform
    _DEFAULT_ODA_PATHS = {
        "Darwin": [
            "/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter",
            "~/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter",
        ],
        "Windows": [
            "C:\\Program Files\\ODA\\ODAFileConverter\\ODAFileConverter.exe",
            "C:\\Program Files (x86)\\ODA\\ODAFileConverter\\ODAFileConverter.exe",
        ],
        "Linux": [
            "/usr/bin/ODAFileConverter",
            "/opt/ODAFileConverter/ODAFileConverter",
            "~/ODAFileConverter/ODAFileConverter",
        ],
    }

    def __init__(self, config: Optional[ConverterConfig] = None):
        """
        Initialize DWG converter.

        Args:
            config: Converter configuration
        """
        self._config = config or ConverterConfig()
        self._oda_path: Optional[str] = None
        self._temp_dir: Optional[Path] = None

        self._find_oda_converter()

    def _find_oda_converter(self) -> None:
        """Find ODA File Converter installation."""
        # Check config path first
        if self._config.oda_path:
            path = Path(self._config.oda_path).expanduser()
            if path.exists():
                self._oda_path = str(path)
                logger.info(f"Using ODA converter at: {self._oda_path}")
                return

        # Check environment variable
        env_path = os.environ.get("ODA_FILE_CONVERTER")
        if env_path:
            path = Path(env_path).expanduser()
            if path.exists():
                self._oda_path = str(path)
                logger.info(f"Using ODA converter from env: {self._oda_path}")
                return

        # Check default paths
        system = platform.system()
        default_paths = self._DEFAULT_ODA_PATHS.get(system, [])

        for path_str in default_paths:
            path = Path(path_str).expanduser()
            if path.exists():
                self._oda_path = str(path)
                logger.info(f"Found ODA converter at: {self._oda_path}")
                return

        logger.warning("ODA File Converter not found. DWG conversion will be limited.")

    @property
    def is_available(self) -> bool:
        """Check if converter is available."""
        return self._oda_path is not None

    @property
    def oda_path(self) -> Optional[str]:
        """Get ODA converter path."""
        return self._oda_path

    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        output_version: Optional[DXFVersion] = None,
    ) -> ConversionResult:
        """
        Convert a DWG file to DXF.

        Args:
            input_path: Path to input DWG file
            output_path: Path for output DXF file (auto-generated if None)
            output_version: DXF version (uses config default if None)

        Returns:
            ConversionResult
        """
        import time

        input_path = Path(input_path)
        start_time = time.time()

        # Validate input
        if not input_path.exists():
            return ConversionResult(
                input_path=str(input_path),
                status=ConversionStatus.FAILED,
                error_message=f"Input file not found: {input_path}",
            )

        if not input_path.suffix.lower() == ".dwg":
            return ConversionResult(
                input_path=str(input_path),
                status=ConversionStatus.SKIPPED,
                error_message="Not a DWG file",
            )

        # Check converter availability
        if not self.is_available:
            return ConversionResult(
                input_path=str(input_path),
                status=ConversionStatus.CONVERTER_NOT_FOUND,
                error_message="ODA File Converter not available",
            )

        # Determine output path
        if output_path is None:
            output_path = input_path.with_suffix(".dxf")
        else:
            output_path = Path(output_path)

        output_version = output_version or self._config.output_version

        try:
            # Get file size before
            file_size_before = input_path.stat().st_size

            # Run conversion
            success = self._run_oda_conversion(input_path, output_path, output_version)

            conversion_time = time.time() - start_time

            if success and output_path.exists():
                file_size_after = output_path.stat().st_size
                return ConversionResult(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    status=ConversionStatus.SUCCESS,
                    conversion_time=conversion_time,
                    file_size_before=file_size_before,
                    file_size_after=file_size_after,
                )
            else:
                return ConversionResult(
                    input_path=str(input_path),
                    status=ConversionStatus.FAILED,
                    error_message="Conversion failed - output file not created",
                    conversion_time=conversion_time,
                    file_size_before=file_size_before,
                )

        except Exception as e:
            logger.error(f"Conversion error: {e}")
            return ConversionResult(
                input_path=str(input_path),
                status=ConversionStatus.FAILED,
                error_message=str(e),
                conversion_time=time.time() - start_time,
            )

    def _run_oda_conversion(
        self,
        input_path: Path,
        output_path: Path,
        output_version: DXFVersion,
    ) -> bool:
        """Run ODA File Converter."""
        # ODA requires input/output directories
        input_dir = input_path.parent
        output_dir = output_path.parent

        # Create temp directory for single file conversion
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input = Path(temp_dir) / "input"
            temp_output = Path(temp_dir) / "output"
            temp_input.mkdir()
            temp_output.mkdir()

            # Copy input file
            temp_input_file = temp_input / input_path.name
            shutil.copy2(input_path, temp_input_file)

            # Build command
            cmd = [
                self._oda_path,
                str(temp_input),
                str(temp_output),
                output_version.value,
                self._config.output_format,
                "0",  # Recursive: 0=no, 1=yes
                "1" if self._config.audit else "0",
            ]

            logger.debug(f"Running ODA converter: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._config.timeout,
                )

                # Check for output file
                expected_output = temp_output / output_path.name
                if not expected_output.exists():
                    # Try with original extension replaced
                    expected_output = temp_output / input_path.with_suffix(".dxf").name

                if expected_output.exists():
                    # Move to final destination
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(expected_output), str(output_path))
                    return True
                else:
                    logger.error(f"ODA output not found. stdout: {result.stdout}, stderr: {result.stderr}")
                    return False

            except subprocess.TimeoutExpired:
                logger.error(f"ODA conversion timeout after {self._config.timeout}s")
                return False
            except Exception as e:
                logger.error(f"ODA conversion error: {e}")
                return False

    def convert_bytes(
        self,
        dwg_bytes: bytes,
        output_version: Optional[DXFVersion] = None,
    ) -> Tuple[Optional[bytes], ConversionResult]:
        """
        Convert DWG bytes to DXF bytes.

        Args:
            dwg_bytes: DWG file content as bytes
            output_version: DXF version

        Returns:
            (dxf_bytes, result)
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / f"{uuid.uuid4().hex}.dwg"
            output_file = temp_path / f"{uuid.uuid4().hex}.dxf"

            # Write input bytes
            input_file.write_bytes(dwg_bytes)

            # Convert
            result = self.convert(input_file, output_file, output_version)

            if result.success and output_file.exists():
                return output_file.read_bytes(), result
            else:
                return None, result

    def get_info(self) -> Dict[str, Any]:
        """Get converter information."""
        return {
            "is_available": self.is_available,
            "oda_path": self._oda_path,
            "output_version": self._config.output_version.value,
            "output_format": self._config.output_format,
        }


def convert_dwg_to_dxf(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    version: DXFVersion = DXFVersion.R2018,
) -> ConversionResult:
    """
    Convenience function to convert a single DWG file.

    Args:
        input_path: Path to DWG file
        output_path: Path for DXF output
        version: DXF version

    Returns:
        ConversionResult
    """
    converter = DWGConverter()
    return converter.convert(input_path, output_path, version)


def batch_convert(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    version: DXFVersion = DXFVersion.R2018,
    recursive: bool = False,
    max_workers: int = 4,
) -> List[ConversionResult]:
    """
    Batch convert DWG files in a directory.

    Args:
        input_dir: Directory containing DWG files
        output_dir: Output directory (same as input if None)
        version: DXF version
        recursive: Process subdirectories
        max_workers: Maximum parallel conversions

    Returns:
        List of ConversionResult
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir

    # Find DWG files
    if recursive:
        dwg_files = list(input_dir.rglob("*.dwg"))
    else:
        dwg_files = list(input_dir.glob("*.dwg"))

    if not dwg_files:
        logger.warning(f"No DWG files found in {input_dir}")
        return []

    logger.info(f"Found {len(dwg_files)} DWG files to convert")

    converter = DWGConverter(ConverterConfig(output_version=version))
    results = []

    def convert_file(dwg_path: Path) -> ConversionResult:
        # Preserve directory structure
        rel_path = dwg_path.relative_to(input_dir)
        out_path = output_dir / rel_path.with_suffix(".dxf")
        return converter.convert(dwg_path, out_path)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(convert_file, f): f for f in dwg_files}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if result.success:
                logger.info(f"Converted: {result.input_path}")
            else:
                logger.error(f"Failed: {result.input_path} - {result.error_message}")

    # Summary
    success_count = sum(1 for r in results if r.success)
    logger.info(f"Conversion complete: {success_count}/{len(results)} successful")

    return results
