from __future__ import annotations

"""Basic MIME validation to protect parsing pipeline.

Uses python-magic if available; degrades gracefully when library absent.
"""

import os
import re
from typing import Any, Dict, Tuple

from fastapi import HTTPException, UploadFile

# Known lightweight CAD format signatures (heuristic) for basic validation.
# This is NOT a full parser; it only checks early markers to catch gross mismatches.
_STEP_SIGNATURE_PREFIX = b"ISO-10303-21"
_STL_ASCII_PREFIX = b"solid"
_IGES_SIGNATURE_TOKENS = [b"IGES", b"S", "G", "D", "P"]  # IGES has section markers (simplified)
_OCR_ALLOWED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
}
_OCR_EXTENSION_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".pdf": "application/pdf",
}
_PDF_FORBIDDEN_TOKENS = (
    b"/JavaScript",
    b"/JS",
    b"/AA",
    b"/OpenAction",
    b"/Launch",
)


def verify_signature(data: bytes, file_format: str) -> Tuple[bool, str]:
    """Best-effort signature / magic validation.

    Returns (valid, expected_hint).
    Only rejects when we are reasonably confident (to avoid false negatives on binary variants).
    """
    header = data[:64]
    fmt = file_format.lower()
    if fmt in {"step", "stp"}:
        return (header.startswith(_STEP_SIGNATURE_PREFIX), "STEP header 'ISO-10303-21'")
    if fmt == "stl":
        # Binary STL starts with 80-byte header; ASCII starts with 'solid'. Accept both by permissive rule.
        if header.startswith(_STL_ASCII_PREFIX):
            return (True, "ASCII STL 'solid'")
        # If not ASCII, assume binary; do not strictly validate beyond length.
        return (len(data) > 84, "Binary STL (>=84 bytes)")
    if fmt in {"iges", "igs"}:
        # IGES often starts with something containing 'IGES' later; lenient check.
        return (b"IGES" in header.upper(), "IGES token present")
    if fmt in {"dxf", "dwg"}:
        # DXF often starts with '0\nSECTION'; DWG binary proprietary; be permissive.
        if b"SECTION" in header.upper() or b"AC101" in header.upper():  # AC101* version tokens
            return (True, "DXF SECTION or DWG AC101*")
        return (True, "Lenient DXF/DWG")  # Do not reject unknown variants
    return (True, "Unknown format lenient")


def signature_hex_prefix(data: bytes, length: int = 16) -> str:
    return data[:length].hex()


def deep_format_validate(data: bytes, file_format: str) -> Tuple[bool, str]:
    """Deep format validation heuristics per CAD format.

    Returns (valid, reason). Non-invasive checks; strict mode decides rejection.
    """
    fmt = file_format.lower()
    head = data[:512]
    if fmt in {"step", "stp"}:
        # STEP should contain HEADER and ENDSEC tokens
        text = head.decode(errors="ignore")
        if "ISO-10303-21" not in text:
            return False, "missing_step_header"
        if "HEADER" not in text:
            return False, "missing_step_HEADER_section"
        return True, "ok"
    if fmt == "stl":
        # Binary STL must be >= 84 bytes; ASCII starts with 'solid'
        if len(data) < 84:
            return False, "stl_too_small"
        if head.startswith(b"solid"):
            return True, "ascii_solid"
        return True, "binary_min_size"
    if fmt in {"iges", "igs"}:
        # IGES sections markers: S,G,D,P,T appear; lenient check just for presence of at least two
        upper = head.upper()
        present = sum(1 for tok in [b"S", b"G", b"D", b"P"] if tok in upper)
        if present < 2:
            return False, "iges_section_markers_missing"
        return True, "ok"
    if fmt == "dxf":
        # DXF typical start: '0\nSECTION'; search for SECTION token
        if b"SECTION" not in head.upper():
            return False, "dxf_section_missing"
        return True, "ok"
    # Other formats lenient
    return True, "ok"


def load_validation_matrix() -> Dict[str, Any]:
    import os

    import yaml

    path = os.getenv("FORMAT_VALIDATION_MATRIX", "config/format_validation_matrix.yaml")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def matrix_validate(
    data: bytes, file_format: str, project_id: str | None = None
) -> Tuple[bool, str]:
    matrix = load_validation_matrix()
    fmts = matrix.get("formats", {})
    spec = fmts.get(file_format.lower())
    if not spec:
        return True, "no_spec"
    head = data[:512].decode(errors="ignore")
    # Exempt projects
    if project_id and project_id in matrix.get("exempt_projects", []):
        return True, "exempt"
    # STEP
    if "required_tokens" in spec:
        for tok in spec["required_tokens"]:
            if tok not in head:
                return False, f"missing_token:{tok}"
    if file_format.lower() == "stl" and spec.get("min_size"):
        if len(data) < int(spec["min_size"]):
            return False, "below_min_size"
    return True, "ok"


def sniff_mime(data: bytes) -> Tuple[str, bool]:
    try:
        import magic  # type: ignore

        mime = magic.from_buffer(data, mime=True) or "application/octet-stream"
        return mime, True
    except Exception:
        return "application/octet-stream", False


def is_supported_mime(mime: str) -> bool:
    # Allow common CAD / text / octet-stream placeholders plus known specific MIME types
    allowed_exact = {
        "text/plain",
        "application/octet-stream",
        "application/zip",  # STEP packages / compressed bundles
        "model/stl",
        "application/stl",
        "application/vnd.ms-pki.stl",
        "application/dxf",
        "image/vnd.dwg",  # some DWG detectors
        "application/iges",
        "application/step",
        "application/x-step",
        "application/x-iges",
    }
    if mime in allowed_exact:
        return True
    # Basic heuristics: many CAD files appear as generic text or octet-stream
    return mime.startswith("text/") or mime.endswith("octet-stream")


__all__ = [
    "sniff_mime",
    "is_supported_mime",
    "verify_signature",
    "deep_format_validate",
    "load_validation_matrix",
    "matrix_validate",
    "validate_and_read",
]


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except ValueError:
        return default


def _resolve_mime(upload_file: UploadFile, data: bytes) -> str:
    sniffed_mime, _ = sniff_mime(data)
    upload_mime = getattr(upload_file, "content_type", None)
    if isinstance(upload_mime, str):
        upload_mime = upload_mime.strip()
    else:
        upload_mime = ""
    if upload_mime and upload_mime != "application/octet-stream":
        return upload_mime
    if sniffed_mime and sniffed_mime != "application/octet-stream":
        return sniffed_mime
    filename = getattr(upload_file, "filename", "") or ""
    if not isinstance(filename, str):
        filename = ""
    ext = os.path.splitext(filename)[1].lower()
    return _OCR_EXTENSION_MIME_MAP.get(ext, "application/octet-stream")


def _count_pdf_pages(data: bytes) -> int:
    text = data.decode(errors="ignore")
    typed_pages = re.findall(r"/Type\\s*/Page\\b", text)
    if typed_pages:
        return len(typed_pages)
    comment_pages = re.findall(r"(?m)^%Page", text)
    if comment_pages:
        return len(comment_pages)
    return len(re.findall(r"/Page(?!s)", text))


def _has_pdf_forbidden_token(data: bytes) -> bool:
    lower = data.lower()
    return any(token.lower() in lower for token in _PDF_FORBIDDEN_TOKENS)


async def validate_and_read(upload_file: UploadFile) -> tuple[bytes, str]:
    """Compatibility helper for OCR module expecting validate_and_read.

    Reads file bytes and returns (data, mime). Keeps logic minimal and reuses sniff_mime.
    """
    data = await upload_file.read()
    max_mb = _get_env_float("OCR_MAX_FILE_MB", 50.0)
    if max_mb > 0 and len(data) > int(max_mb * 1024 * 1024):
        raise HTTPException(status_code=413, detail="File too large")
    mime = _resolve_mime(upload_file, data)
    ext = os.path.splitext(upload_file.filename or "")[1].lower()
    is_pdf = data.startswith(b"%PDF")
    is_image = mime.startswith("image/") or ext in _OCR_ALLOWED_IMAGE_EXTENSIONS
    if not is_pdf and mime == "application/pdf":
        is_pdf = True
    if not (is_pdf or is_image):
        raise HTTPException(status_code=415, detail="Unsupported MIME type")
    if is_pdf:
        max_pages = _get_env_int("OCR_MAX_PDF_PAGES", 20)
        page_count = _count_pdf_pages(data)
        if page_count and page_count > max_pages:
            raise HTTPException(status_code=400, detail="PDF page count exceeded")
        if _has_pdf_forbidden_token(data):
            raise HTTPException(status_code=400, detail="PDF forbidden token detected")
    return data, mime
