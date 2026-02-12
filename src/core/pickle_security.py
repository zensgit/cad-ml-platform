"""Pickle security scanner with audit, blocklist, and whitelist modes.

This module provides security scanning for pickle files before loading them,
helping prevent deserialization attacks through dangerous opcodes.

Modes:
- audit: Logs all opcodes but never blocks - for observation/learning
- blocklist: Blocks known dangerous opcodes (default production mode)
- whitelist: Only allows explicitly safe opcodes (strictest mode)

Usage:
    from src.core.pickle_security import scan_pickle_opcodes, OpcodeMode

    result = scan_pickle_opcodes(file_path, mode=OpcodeMode.BLOCKLIST)
    if not result["safe"]:
        raise SecurityError(f"Dangerous opcodes: {result['dangerous']}")
"""

from __future__ import annotations

import logging
import os
import pickletools
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class OpcodeMode(str, Enum):
    """Pickle opcode security modes."""

    AUDIT = "audit"
    BLOCKLIST = "blocklist"
    WHITELIST = "whitelist"


# Known dangerous opcodes that can enable code execution
DANGEROUS_OPCODES: Set[str] = {
    "GLOBAL",  # Load global object (module.attr)
    "STACK_GLOBAL",  # Load global from stack
    "REDUCE",  # Apply callable to args (main RCE vector)
    "INST",  # Build and push class instance
    "OBJ",  # Build object by calling class
    "NEWOBJ",  # Build object using cls.__new__
    "NEWOBJ_EX",  # Extended NEWOBJ
    "BUILD",  # Call obj.__setstate__ or update __dict__
    "EXT1",  # Extension registry lookups
    "EXT2",
    "EXT4",
}

# Safe opcodes allowed in whitelist mode (minimal safe set)
SAFE_OPCODES: Set[str] = {
    # Basic types
    "NONE",
    "NEWTRUE",
    "NEWFALSE",
    "INT",
    "BININT",
    "BININT1",
    "BININT2",
    "LONG",
    "LONG1",
    "LONG4",
    "FLOAT",
    "BINFLOAT",
    # Strings
    "STRING",
    "BINSTRING",
    "SHORT_BINSTRING",
    "UNICODE",
    "BINUNICODE",
    "SHORT_BINUNICODE",
    "BINUNICODE8",
    "BINBYTES",
    "SHORT_BINBYTES",
    "BINBYTES8",
    "BYTEARRAY8",
    # Collections
    "EMPTY_LIST",
    "APPEND",
    "APPENDS",
    "LIST",
    "EMPTY_TUPLE",
    "TUPLE",
    "TUPLE1",
    "TUPLE2",
    "TUPLE3",
    "EMPTY_DICT",
    "DICT",
    "SETITEM",
    "SETITEMS",
    "EMPTY_SET",
    "ADDITEMS",
    "FROZENSET",
    # Memo operations
    "PUT",
    "BINPUT",
    "LONG_BINPUT",
    "GET",
    "BINGET",
    "LONG_BINGET",
    "MEMOIZE",
    # Control
    "MARK",
    "POP",
    "POP_MARK",
    "DUP",
    "STOP",
    # Protocol markers
    "PROTO",
    "FRAME",
}


def scan_pickle_opcodes(
    source: Union[str, Path, bytes, BinaryIO],
    mode: OpcodeMode = OpcodeMode.BLOCKLIST,
    include_positions: bool = False,
) -> Dict[str, Any]:
    """Scan pickle data for opcodes and assess safety.

    Args:
        source: File path, bytes, or file-like object containing pickle data
        mode: Security mode (audit, blocklist, whitelist)
        include_positions: Include position information for each opcode

    Returns:
        Dictionary with scan results:
        - safe: bool - Whether the pickle is considered safe
        - mode: str - Mode used for scanning
        - opcodes: List[str] - All opcodes found
        - opcode_counts: Dict[str, int] - Count of each opcode
        - dangerous: List[str] - Dangerous opcodes found (blocklist mode)
        - disallowed: List[str] - Disallowed opcodes (whitelist mode)
        - positions: List[Dict] - Position info (if include_positions=True)
        - blocked_reason: Optional[str] - Reason if blocked
    """
    # Get data
    data = _get_pickle_data(source)

    # Initialize result
    result: Dict[str, Any] = {
        "safe": True,
        "mode": mode.value,
        "opcodes": [],
        "opcode_counts": {},
        "dangerous": [],
        "disallowed": [],
        "blocked_reason": None,
    }

    if include_positions:
        result["positions"] = []

    try:
        for op, arg, pos in pickletools.genops(data):
            opcode_name = op.name

            # Record opcode
            result["opcodes"].append(opcode_name)
            result["opcode_counts"][opcode_name] = result["opcode_counts"].get(opcode_name, 0) + 1

            if include_positions:
                result["positions"].append(
                    {
                        "opcode": opcode_name,
                        "position": pos,
                        "arg": str(arg) if arg is not None else None,
                    }
                )

            # Check based on mode
            if mode == OpcodeMode.BLOCKLIST:
                if opcode_name in DANGEROUS_OPCODES:
                    result["dangerous"].append(opcode_name)
                    result["safe"] = False
                    result["blocked_reason"] = f"Dangerous opcode '{opcode_name}' at position {pos}"

            elif mode == OpcodeMode.WHITELIST:
                if opcode_name not in SAFE_OPCODES:
                    result["disallowed"].append(opcode_name)
                    result["safe"] = False
                    result[
                        "blocked_reason"
                    ] = f"Disallowed opcode '{opcode_name}' at position {pos}"

            # Audit mode never blocks
            # (safe remains True)

        # Deduplicate lists
        result["dangerous"] = sorted(set(result["dangerous"]))
        result["disallowed"] = sorted(set(result["disallowed"]))

    except Exception as e:
        result["safe"] = False
        result["blocked_reason"] = f"Scan error: {str(e)}"
        result["scan_error"] = str(e)

    return result


def _get_pickle_data(source: Union[str, Path, bytes, BinaryIO]) -> bytes:
    """Extract pickle data from various source types."""
    if isinstance(source, bytes):
        return source
    elif isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            return f.read()
    else:
        # File-like object
        return source.read()


def validate_pickle_file(
    file_path: Union[str, Path],
    mode: Optional[OpcodeMode] = None,
) -> Dict[str, Any]:
    """Validate a pickle file for security.

    Uses environment variable MODEL_OPCODE_MODE if mode not specified.

    Args:
        file_path: Path to pickle file
        mode: Security mode (defaults to env var or blocklist)

    Returns:
        Validation result dictionary
    """
    if mode is None:
        mode_str = os.getenv("MODEL_OPCODE_MODE", "blocklist")
        try:
            mode = OpcodeMode(mode_str)
        except ValueError:
            mode = OpcodeMode.BLOCKLIST

    path = Path(file_path)
    result = {
        "path": str(path),
        "exists": path.exists(),
        "mode": mode.value,
    }

    if not path.exists():
        result["safe"] = False
        result["error"] = "File not found"
        return result

    # Check file size
    max_mb = float(os.getenv("MODEL_MAX_MB", "50"))
    size_mb = path.stat().st_size / (1024 * 1024)
    result["size_mb"] = round(size_mb, 3)

    if size_mb > max_mb:
        result["safe"] = False
        result["error"] = f"File size {size_mb:.1f}MB exceeds limit {max_mb}MB"
        return result

    # Scan opcodes
    scan_result = scan_pickle_opcodes(path, mode=mode)
    result.update(scan_result)

    return result


def get_opcode_mode_from_env() -> OpcodeMode:
    """Get the current opcode mode from environment."""
    mode_str = os.getenv("MODEL_OPCODE_MODE", "blocklist")
    try:
        return OpcodeMode(mode_str)
    except ValueError:
        return OpcodeMode.BLOCKLIST


def get_security_config() -> Dict[str, Any]:
    """Get current pickle security configuration."""
    return {
        "opcode_mode": get_opcode_mode_from_env().value,
        "opcode_scan_enabled": os.getenv("MODEL_OPCODE_SCAN", "1") == "1",
        "opcode_strict": os.getenv("MODEL_OPCODE_STRICT", "0") == "1",
        "magic_number_check": os.getenv("MODEL_MAGIC_NUMBER_CHECK", "1") == "1",
        "max_model_size_mb": float(os.getenv("MODEL_MAX_MB", "50")),
        "hash_whitelist_enabled": bool(os.getenv("ALLOWED_MODEL_HASHES", "").strip()),
        "dangerous_opcodes": sorted(DANGEROUS_OPCODES),
        "safe_opcodes_count": len(SAFE_OPCODES),
    }


def audit_pickle_directory(
    directory: Union[str, Path],
    pattern: str = "*.pkl",
) -> Dict[str, Any]:
    """Audit all pickle files in a directory.

    Useful for auditing existing models before enabling stricter modes.

    Args:
        directory: Directory to scan
        pattern: Glob pattern for pickle files

    Returns:
        Audit summary with per-file results
    """
    path = Path(directory)
    files = list(path.glob(pattern))

    files_safe: int = 0
    files_unsafe: int = 0
    all_opcodes: Dict[str, int] = {}
    dangerous_found: Set[str] = set()
    file_results: List[Dict[str, Any]] = []

    for file_path in files:
        scan = scan_pickle_opcodes(file_path, mode=OpcodeMode.BLOCKLIST)

        if scan["safe"]:
            files_safe += 1
        else:
            files_unsafe += 1
            dangerous_found.update(scan["dangerous"])

        # Aggregate opcode counts
        for opcode, count in scan["opcode_counts"].items():
            all_opcodes[opcode] = all_opcodes.get(opcode, 0) + count

        file_results.append(
            {
                "file": str(file_path.name),
                "safe": scan["safe"],
                "opcode_count": len(scan["opcodes"]),
                "dangerous": scan["dangerous"],
            }
        )

    return {
        "directory": str(path),
        "pattern": pattern,
        "files_scanned": len(files),
        "files_safe": files_safe,
        "files_unsafe": files_unsafe,
        "all_opcodes": all_opcodes,
        "dangerous_found": sorted(dangerous_found),
        "file_results": file_results,
    }


__all__ = [
    "OpcodeMode",
    "DANGEROUS_OPCODES",
    "SAFE_OPCODES",
    "scan_pickle_opcodes",
    "validate_pickle_file",
    "get_opcode_mode_from_env",
    "get_security_config",
    "audit_pickle_directory",
]
