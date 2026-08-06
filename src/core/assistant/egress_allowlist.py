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
        "citable",
        "disclosure",
        # Hashed / opaque ids only when explicitly named as such:
        "candidate_id_hash",
        "file_id_hash",
    }
)

_PATH_HINT = re.compile(
    r"(?:[A-Za-z]:\\|/(?:Users|home|var|tmp|data|models)/|\\\\)",
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


# Prompt-assembly keys permitted for hosted chat APIs (text only).
HOSTED_PROMPT_FIELDS = frozenset(
    {
        "system_prompt",
        "user_prompt",
        "role",
        "content",
        "messages",
        "type",
        "text",
    }
)

# Substrings that must never appear in hosted free-text prompts.
_FORBIDDEN_CONTENT_MARKERS = (
    "ocr_text",
    "raw_ocr",
    "drawing_bytes",
    "image_bytes",
    "file_path=",
    "filepath=",
)


def _scan_text_content(text: str, *, where: str) -> None:
    if not isinstance(text, str):
        raise EgressRejected(f"non_text_{where}")
    if _PATH_HINT.search(text):
        raise EgressRejected(f"path_like_{where}")
    if _BASE64_LONG.search(text):
        # Long base64 blobs often encode images/files — refuse.
        raise EgressRejected(f"base64_blob_{where}")
    lower = text.lower()
    for marker in _FORBIDDEN_CONTENT_MARKERS:
        if marker in lower:
            raise EgressRejected(f"forbidden_marker_{marker}")


def enforce_hosted_prompt_egress(system_prompt: str, user_prompt: str) -> None:
    """Gate free-text prompts immediately before a hosted-provider network call.

    This is the live-path enforcement for §2.B when the call site sends chat
    prompts rather than a structured tool JSON object.
    """
    if isinstance(system_prompt, (bytes, bytearray)) or isinstance(
        user_prompt, (bytes, bytearray)
    ):
        raise EgressRejected("raw_bytes_prompt")
    _scan_text_content(system_prompt or "", where="system_prompt")
    _scan_text_content(user_prompt or "", where="user_prompt")
    # Also validate as a structured envelope so field-name bans apply if
    # callers later pass dict-shaped content through the same helper.
    validate_hosted_payload(
        {
            "system_prompt": system_prompt or "",
            "user_prompt": user_prompt or "",
        },
        allowed_fields=set(HOSTED_PROMPT_FIELDS) | set(DEFAULT_ALLOWED_FIELDS),
    )


def enforce_tool_result_for_hosted(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Redact a tool result before it may be sent to a hosted model.

    Non-citable results are reduced to status/reason_code only so fabricated
    business fields never leave the process toward a third party.
    """
    from src.core.assistant.tool_status import is_citable_tool_result

    if not is_citable_tool_result(result):
        status = result.get("status", "failed")
        reason = result.get("reason_code", "non_citable")
        slim = {"status": status, "reason_code": reason, "citable": False}
        validate_hosted_payload(slim)
        return slim
    # Citable structured results still pass the field allowlist (strict).
    try:
        validate_hosted_payload(result)
        return dict(result)
    except EgressRejected:
        # Drop to scores/status only rather than aborting the whole turn when a
        # tool returns extra keys — still fail closed on forbidden categories.
        slim = {
            k: v
            for k, v in result.items()
            if str(k).lower()
            in DEFAULT_ALLOWED_FIELDS | {"citable", "status", "reason_code", "label"}
        }
        validate_hosted_payload(slim)
        return slim


__all__ = [
    "DEFAULT_ALLOWED_FIELDS",
    "EgressRejected",
    "FORBIDDEN_FIELD_NAMES",
    "HOSTED_PROMPT_FIELDS",
    "enforce_hosted_prompt_egress",
    "enforce_tool_result_for_hosted",
    "redact_for_hosted",
    "validate_hosted_payload",
]
