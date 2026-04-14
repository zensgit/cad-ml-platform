"""DXF text content extraction for B5.1 text-fusion classifier.

Extracts all human-readable text from DXF modelspace entities:
TEXT, MTEXT, ATTRIB, ATTDEF.

Handles two common DXF text encodings:
  - UTF-8 / native Chinese (e.g. AutoCAD 2018+)
  - \\M+5XXXX GB2312/GBK multibyte escape sequences (older CAD tools)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# GB2312 / GBK multibyte escape: \M+5<hex> (codepage 5 = GBK)
_M_ESCAPE = re.compile(r"\\M\+5([0-9A-Fa-f]+)")
# MTEXT inline format codes: \C2; \H1.5; \A1; etc. (backslash + letters/digits + semicolon)
_MTEXT_INLINE = re.compile(r"\\[A-Za-z][^;\\{}\n]*;?")
# Paragraph break: \P or \n
_PARA_BREAK = re.compile(r"\\[Pp]|\r")


def _decode_m_escapes(text: str) -> str:
    """Decode \\M+5<hex> GBK byte sequences to Unicode characters."""
    def repl(m: re.Match) -> str:
        hex_str = m.group(1)
        try:
            raw = bytes.fromhex(hex_str)
            return raw.decode("gbk", errors="replace")
        except Exception:
            return ""
    return _M_ESCAPE.sub(repl, text)


def _clean_mtext(raw: str) -> str:
    """Return plain readable text from a raw MTEXT string.

    Processing order matters:
      1. Decode \M+5<hex> GBK sequences FIRST (before stripping backslash codes)
      2. Replace paragraph breaks (\P) with spaces
      3. Strip remaining inline format codes (\C2; \H1.5; etc.)
      4. Strip curly braces used as grouping (not content delimiters)
      5. Collapse whitespace
    """
    # 1. Decode GBK escape sequences (must be first — uses backslash syntax)
    text = _decode_m_escapes(raw)
    # 2. Paragraph breaks → space
    text = _PARA_BREAK.sub(" ", text)
    # 3. Strip inline control codes (backslash + letter + params + semicolon)
    text = _MTEXT_INLINE.sub("", text)
    # 4. Strip bare curly braces (grouping delimiters, not content)
    text = text.replace("{", "").replace("}", "")
    # 5. Collapse whitespace
    text = " ".join(text.split())
    return text


def extract_text_from_path(dxf_path: str) -> str:
    """Extract all visible text from a DXF file at the given path.

    Returns:
        Single string of all extracted text, space-joined. Empty on failure.
    """
    try:
        import ezdxf
        doc = ezdxf.readfile(str(dxf_path))
        return _extract_from_doc(doc)
    except Exception as exc:
        logger.debug("text_extractor: failed to read %s — %s", dxf_path, exc)
        return ""


def extract_text_from_bytes(dxf_bytes: bytes) -> str:
    """Extract all visible text from raw DXF bytes.

    Returns:
        Single string of all extracted text. Empty on failure.
    """
    try:
        import ezdxf
        import io
        doc = ezdxf.read(io.BytesIO(dxf_bytes))
        return _extract_from_doc(doc)
    except Exception as exc:
        logger.debug("text_extractor: failed to parse bytes — %s", exc)
        return ""


def _extract_from_doc(doc) -> str:
    """Internal: extract text from an already-parsed ezdxf document."""
    texts: list[str] = []

    try:
        msp = doc.modelspace()
    except Exception:
        return ""

    for entity in msp:
        try:
            t = _entity_text(entity)
            if t:
                texts.append(t)
        except Exception:
            continue

    # Also scan block definitions for attribute text
    try:
        for block in doc.blocks:
            for entity in block:
                try:
                    if entity.dxftype() in ("ATTDEF", "ATTRIB"):
                        t = _entity_text(entity)
                        if t:
                            texts.append(t)
                except Exception:
                    continue
    except Exception:
        pass

    return " ".join(texts)


def _entity_text(entity) -> Optional[str]:
    """Extract and clean text from a single DXF entity."""
    etype = entity.dxftype()

    if etype == "MTEXT":
        # Try plain_text() first (ezdxf 1.x API)
        raw = ""
        try:
            raw = entity.plain_text()
        except Exception:
            pass
        if not raw or "\\M+" in raw:
            # Fall back to raw dxf.text with manual decode
            try:
                raw = entity.dxf.text or ""
                raw = _clean_mtext(raw)
            except Exception:
                raw = ""
        else:
            raw = " ".join(raw.split())
        return raw.strip() or None

    if etype in ("TEXT", "ATTRIB", "ATTDEF"):
        try:
            t = entity.dxf.get("text", "").strip()
            if not t:
                return None
            # Decode GBK \M+5 escapes if present
            if "M+" in t:
                t = _decode_m_escapes(t)
            return t.strip() or None
        except Exception:
            return None

    return None
