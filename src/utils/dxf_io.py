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
from typing import Any, Iterable, List, Optional, Set

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

    import ezdxf

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


def write_dxf_document_to_bytes(doc: Any, encoding: str = "utf-8") -> bytes:
    """Write an ezdxf document to bytes.

    Note: ``ezdxf`` writes strings to the stream for ASCII DXF output, so we
    use a text wrapper around a BytesIO buffer.
    """

    if not encoding:
        encoding = "utf-8"

    buf = io.BytesIO()
    wrapper = io.TextIOWrapper(
        buf,
        encoding=encoding,
        errors="ignore",
        newline="\n",
    )
    try:
        doc.write(wrapper)
        wrapper.flush()
    finally:
        # Detach so the underlying BytesIO stays readable.
        try:
            wrapper.detach()
        except Exception:  # noqa: BLE001
            pass
    return buf.getvalue()


def strip_dxf_entities_from_bytes(
    data: bytes,
    dxftypes: Iterable[str],
    *,
    strip_blocks: bool = True,
) -> bytes:
    """Remove selected entity dxftypes from a DXF payload and return new bytes.

    This is intended for local evaluation / data sanitization workflows, for
    example to simulate geometry-only inference by removing annotation entities.
    """

    if not data:
        raise ValueError("DXF bytes cannot be empty")

    wanted: Set[str] = {str(t).strip().upper() for t in dxftypes if str(t).strip()}
    if not wanted:
        return data

    doc = read_dxf_document_from_bytes(data)

    def _strip_layout(layout: Any) -> None:
        try:
            entities = list(layout)
        except Exception:  # noqa: BLE001
            return
        for ent in entities:
            try:
                if str(ent.dxftype() or "").upper() not in wanted:
                    continue
                layout.delete_entity(ent)
            except Exception:  # noqa: BLE001
                # Best-effort: stripping is only used for evaluation workflows.
                continue

    _strip_layout(doc.modelspace())

    if strip_blocks:
        try:
            for block in doc.blocks:  # type: ignore[attr-defined]
                _strip_layout(block)
        except Exception:  # noqa: BLE001
            pass

    encoding = guess_dxf_encoding(data)
    return write_dxf_document_to_bytes(doc, encoding=encoding)


def strip_dxf_text_entities_from_bytes(
    data: bytes,
    *,
    strip_blocks: bool = True,
) -> bytes:
    """Remove common text/annotation entities from a DXF payload.

    This includes modelspace and (optionally) block definitions to avoid leaking
    titleblock text via INSERT virtual entities.
    """

    text_types = {
        "TEXT",
        "MTEXT",
        "DIMENSION",
        "ATTRIB",
        "ATTDEF",
    }
    return strip_dxf_entities_from_bytes(
        data,
        text_types,
        strip_blocks=strip_blocks,
    )
