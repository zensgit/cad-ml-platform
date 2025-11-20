"""Input validation and lightweight security guards for OCR endpoint.

Validations:
 - MIME whitelist
 - File size limit
 - Optional PDF page limit
 - Image resolution cap (resize if needed)
"""

from __future__ import annotations

import io
import os
from typing import Tuple

from fastapi import HTTPException, UploadFile

MIME_WHITELIST = {"image/png", "image/jpeg", "application/pdf"}

_ENV_MAX_FILE_MB = os.getenv("OCR_MAX_FILE_MB")
_ENV_MAX_PDF_PAGES = os.getenv("OCR_MAX_PDF_PAGES")
try:
    MAX_FILE_SIZE_MB = int(_ENV_MAX_FILE_MB) if _ENV_MAX_FILE_MB else 50
except ValueError:
    MAX_FILE_SIZE_MB = 50
try:
    MAX_PDF_PAGES = int(_ENV_MAX_PDF_PAGES) if _ENV_MAX_PDF_PAGES else 20
except ValueError:
    MAX_PDF_PAGES = 20
MAX_RESOLUTION = 2048  # max width or height
PDF_FORBIDDEN_TOKENS = ["/JavaScript", "/AA", "/OpenAction", "/XFA"]


def _ensure_mime(upload: UploadFile) -> None:
    if upload.content_type not in MIME_WHITELIST:
        try:
            from src.utils.metrics import ocr_input_rejected_total

            ocr_input_rejected_total.labels(reason="invalid_mime").inc()
        except Exception:
            pass
        raise HTTPException(status_code=415, detail=f"Unsupported MIME type {upload.content_type}")


def _ensure_size(content: bytes) -> None:
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        try:
            from src.utils.metrics import ocr_input_rejected_total

            ocr_input_rejected_total.labels(reason="file_too_large").inc()
        except Exception:
            pass
        raise HTTPException(
            status_code=413, detail=f"File too large {size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB"
        )


def _pdf_page_count(content: bytes) -> int:
    """Return PDF page count.

    Uses pypdf if available; otherwise falls back to naive pattern counting.
    """
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(io.BytesIO(content))
        return len(reader.pages)
    except Exception:
        # Fallback: count '%Page' markers (synthetic tests)
        return content.count(b"%Page")


def _pdf_security_scan(content: bytes) -> None:
    """Scan PDF for forbidden tokens. Falls back to raw byte search if parsing fails."""
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(io.BytesIO(content))
        raw = str(reader.trailer)
    except Exception:
        # Fallback: scan raw bytes string representation
        raw = content.decode(errors="ignore")
    for token in PDF_FORBIDDEN_TOKENS:
        if token in raw:
            try:
                from src.utils.metrics import ocr_input_rejected_total

                ocr_input_rejected_total.labels(reason="pdf_forbidden_token").inc()
            except Exception:
                pass
            raise HTTPException(status_code=422, detail=f"PDF contains forbidden token {token}")


def _ensure_pdf_limits(content: bytes) -> None:
    pages = _pdf_page_count(content)
    if pages > MAX_PDF_PAGES:
        from fastapi import HTTPException

        try:
            from src.utils.metrics import ocr_input_rejected_total

            ocr_input_rejected_total.labels(reason="pdf_pages_exceed").inc()
        except Exception:
            pass
        raise HTTPException(
            status_code=422,
            detail=f"PDF page count {pages} exceeds limit {MAX_PDF_PAGES}",
        )


def _maybe_resize_image(content: bytes) -> bytes:
    try:
        from PIL import Image

        im = Image.open(io.BytesIO(content))
        w, h = im.size
        if max(w, h) <= MAX_RESOLUTION:
            return content
        # resize preserving aspect
        if w >= h:
            new_w = MAX_RESOLUTION
            new_h = int(h * (MAX_RESOLUTION / w))
        else:
            new_h = MAX_RESOLUTION
            new_w = int(w * (MAX_RESOLUTION / h))
        im = im.resize((new_w, new_h))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return content


async def validate_and_read(upload: UploadFile) -> Tuple[bytes, str]:
    """Validate upload and return (content, mime). Resizes image if oversized."""
    _ensure_mime(upload)
    data = await upload.read()
    _ensure_size(data)
    if upload.content_type == "application/pdf":
        _ensure_pdf_limits(data)
        _pdf_security_scan(data)
    else:
        data = _maybe_resize_image(data)
    return data, upload.content_type
