from __future__ import annotations

"""Basic MIME validation to protect parsing pipeline.

Uses python-magic if available; degrades gracefully when library absent.
"""

from typing import Tuple, Dict, Any

# Known lightweight CAD format signatures (heuristic) for basic validation.
# This is NOT a full parser; it only checks early markers to catch gross mismatches.
_STEP_SIGNATURE_PREFIX = b"ISO-10303-21"
_STL_ASCII_PREFIX = b"solid"
_IGES_SIGNATURE_TOKENS = [b"IGES", b"S","G","D","P"]  # IGES has section markers (simplified)


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
    import os, yaml
    path = os.getenv("FORMAT_VALIDATION_MATRIX", "config/format_validation_matrix.yaml")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def matrix_validate(data: bytes, file_format: str, project_id: str | None = None) -> Tuple[bool, str]:
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


async def validate_and_read(upload_file) -> tuple[bytes, str]:  # type: ignore
    """Compatibility helper for OCR module expecting validate_and_read.

    Reads file bytes and returns (data, mime). Keeps logic minimal and reuses sniff_mime.
    """
    data = await upload_file.read()
    mime, _ = sniff_mime(data)
    return data, mime
