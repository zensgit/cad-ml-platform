"""Field-level allowlist for hosted-LLM payloads (§2.B).

Rejects payloads that carry raw drawing bytes, OCR text, filesystem paths,
or non-allowlisted identifier fields **before** any network call.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

# Fields always forbidden in a hosted payload (category ban).
FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "raw_bytes",
        "image_bytes",
        "drawing_bytes",
        "file_bytes",
        "ocr_text",
        "raw_ocr",
        "ocr_raw",
        "file_path",
        "filepath",
        "path",
        "absolute_path",
        "local_path",
        "supplier",
        "supplier_name",
        "drawing_number",
        "part_number",
        "dwg_no",
        "process_notes",
        "material_notes",
        "free_text",
        "customer_text",
    }
)

# Default allowlist for explainability-only fields (scores / categories).
DEFAULT_ALLOWED_FIELDS = frozenset(
    {
        "score",
        "scores",
        "confidence",
        "similarity",
        "rejection_reason_category",
        "rejection_category",
        "label",
        "status",
        "reason_code",
        "decision",
        "rank",
        "count",
        # Hashed / opaque ids only when explicitly named as such:
        "candidate_id_hash",
        "file_id_hash",
    }
)

_PATH_HINT = re.compile(
    r"(^|/)([A-Za-z]:\\|/(?:Users|home|var|tmp|data|models)/|\\\\)",
    re.IGNORECASE,
)
_BASE64_LONG = re.compile(r"[A-Za-z0-9+/]{80,}={0,2}")


class EgressRejected(Exception):
    """Payload failed the hosted-egress allowlist."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _iter_paths(obj: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            yield key, v
            yield from _iter_paths(v, key)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            yield key, v
            yield from _iter_paths(v, key)


def validate_hosted_payload(
    payload: Any,
    *,
    allowed_fields: Optional[Set[str]] = None,
) -> None:
    """Raise :class:`EgressRejected` if *payload* is not safe for hosted LLM.

    Rules (design-lock §2.B):
    - Forbidden field names are always rejected.
    - Leaf keys must be in the allowlist (default explainability set).
    - Values must not look like raw base64 blobs, filesystem paths, or bytes.
    """
    if payload is None:
        return
    allow = set(allowed_fields or DEFAULT_ALLOWED_FIELDS)

    if isinstance(payload, (bytes, bytearray)):
        raise EgressRejected("raw_bytes_payload")

    for path, value in _iter_paths(payload):
        leaf = path.split(".")[-1].split("[")[0].lower()
        if leaf in FORBIDDEN_FIELD_NAMES:
            raise EgressRejected(f"forbidden_field:{leaf}")
        # Allowlist applies to leaf object keys only (not array indices).
        if "[" not in path.split(".")[-1] and leaf and leaf not in allow:
            # Nested dict keys that are structural containers are ok if they only
            # hold allowlisted children — reject unknown leaf keys.
            if not isinstance(value, (Mapping, list, tuple)):
                raise EgressRejected(f"non_allowlisted_field:{leaf}")
        if isinstance(value, (bytes, bytearray)):
            raise EgressRejected("raw_bytes_value")
        if isinstance(value, str):
            if _PATH_HINT.search(value):
                raise EgressRejected("path_like_value")
            if _BASE64_LONG.fullmatch(value.strip()):
                raise EgressRejected("raw_base64_blob")


def redact_for_hosted(
    payload: Mapping[str, Any],
    *,
    allowed_fields: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Return a shallow copy keeping only allowlisted top-level fields.

    Still runs :func:`validate_hosted_payload` on the result.
    """
    allow = set(allowed_fields or DEFAULT_ALLOWED_FIELDS)
    out = {k: v for k, v in payload.items() if str(k).lower() in allow}
    validate_hosted_payload(out, allowed_fields=allow)
    return out


__all__ = [
    "DEFAULT_ALLOWED_FIELDS",
    "EgressRejected",
    "FORBIDDEN_FIELD_NAMES",
    "redact_for_hosted",
    "validate_hosted_payload",
]
