"""DXF I/O utilities.

Why: several call sites receive DXF bytes (FastAPI upload, TestClient, etc.)
and previously round-tripped through a temp file to use ``ezdxf.readfile``.
That adds I/O overhead and can be noisy in restricted environments.

This module provides a small helper to read a DXF document directly from bytes
with a best-effort encoding guess based on the header.
"""

from __future__ import annotations

import io
import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_HEADER_SCAN_BYTES = 128 * 1024


def _find_header_value(lines: List[str], key: str) -> Optional[str]:
    """Return the value line for a DXF header key (e.g. ``$ACADVER``)."""

    needle = key.strip()
    for idx, line in enumerate(lines):
        if line.strip() != needle:
            continue
        # DXF header layout:
        #   9
        #   $ACADVER
        #   1
        #   AC1032
        value_idx = idx + 2
        if value_idx >= len(lines):
            return None
        value = lines[value_idx].strip()
        return value or None
    return None


def guess_dxf_encoding(data: bytes) -> str:
    """Guess the DXF text encoding from the header (best effort).

    - For DXF R2007+ (AC1021 and later), DXF text is UTF-8.
    - For older versions, fall back to $DWGCODEPAGE when present.
    """

    if not data:
        return "utf-8"

    header_text = data[:_HEADER_SCAN_BYTES].decode("latin1", errors="ignore")
    lines = header_text.splitlines()

    acadver = _find_header_value(lines, "$ACADVER")
    if acadver and acadver >= "AC1021":
        return "utf-8"

    codepage = _find_header_value(lines, "$DWGCODEPAGE")
    if codepage:
        try:
            from ezdxf.tools import codepage as ezdxf_codepage

            encoding = ezdxf_codepage.toencoding(codepage)
            if encoding:
                return encoding
        except Exception:  # noqa: BLE001
            logger.debug("Failed to map DXF codepage to encoding", exc_info=True)

    return "utf-8"


def read_dxf_document_from_bytes(data: bytes) -> Any:
    """Read a DXF document from bytes using ``ezdxf.read``."""

    if not data:
        raise ValueError("DXF bytes cannot be empty")

    import ezdxf  # type: ignore

    encoding = guess_dxf_encoding(data)
    with io.TextIOWrapper(
        io.BytesIO(data),
        encoding=encoding,
        errors="ignore",
        newline=None,
    ) as stream:
        return ezdxf.read(stream)


def read_dxf_entities_from_bytes(data: bytes) -> List[Any]:
    """Convenience wrapper returning ``list(doc.modelspace())`` from DXF bytes."""

    doc = read_dxf_document_from_bytes(data)
    return list(doc.modelspace())

