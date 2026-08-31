"""Canonical JSON helpers for ReviewReuse ledger digests."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List


class CanonicalJSONError(ValueError):
    """Raised when a value cannot be represented as RFC 8785 I-JSON."""


def canonical_json_v1(value: Any) -> bytes:
    """Return an RFC 8785-compatible UTF-8 representation.

    ReviewReuse digest payloads are composed of JSON primitives. Python's
    shortest-round-trip float representation uses the same significant digits
    as ECMAScript; ``_number`` normalizes its decimal/exponent presentation to
    the RFC 8785 thresholds.
    """

    return _encode(value).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_v1(value)).hexdigest()


def strict_json_loads(payload: str | bytes) -> Any:
    """Parse one I-JSON value without accepting duplicate object keys."""

    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        canonical_json_v1(value)
    except (CanonicalJSONError, TypeError, UnicodeError, ValueError) as exc:
        if isinstance(exc, CanonicalJSONError):
            raise
        raise CanonicalJSONError("invalid I-JSON payload") from exc
    return value


def _reject_constant(value: str) -> Any:
    raise CanonicalJSONError(f"non-finite JSON number: {value}")


def _unique_object(pairs: List[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJSONError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _encode(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        _validate_string(value)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, int):
        if abs(value) > 9_007_199_254_740_991:
            raise CanonicalJSONError("integer exceeds the I-JSON safe range")
        return str(value)
    if isinstance(value, float):
        return _number(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_encode(item) for item in value) + "]"
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise CanonicalJSONError("object keys must be strings")
        for key in value:
            _validate_string(key)
        keys = sorted(value, key=lambda key: key.encode("utf-16be"))
        return (
            "{"
            + ",".join(f"{_encode(key)}:{_encode(value[key])}" for key in keys)
            + "}"
        )
    raise CanonicalJSONError(f"unsupported JSON value: {type(value).__name__}")


def _validate_string(value: str) -> None:
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise CanonicalJSONError("lone UTF-16 surrogate is not valid I-JSON")


def _number(value: float) -> str:
    if not math.isfinite(value):
        raise CanonicalJSONError("non-finite number is not valid I-JSON")
    if value == 0:
        return "0"

    negative = value < 0
    raw = repr(abs(value)).lower()
    if "e" in raw:
        coefficient, exponent_text = raw.split("e", 1)
        exponent = int(exponent_text)
    else:
        coefficient = raw
        exponent = 0

    if "." in coefficient:
        integer, fraction = coefficient.split(".", 1)
    else:
        integer, fraction = coefficient, ""
    if fraction == "0":
        fraction = ""
    digits = (integer + fraction).lstrip("0") or "0"
    decimal_position = len(integer) + exponent - (len(integer + fraction) - len(digits))

    if len(digits) <= decimal_position <= 21:
        encoded = digits + ("0" * (decimal_position - len(digits)))
    elif 0 < decimal_position <= 21:
        encoded = digits[:decimal_position] + "." + digits[decimal_position:]
    elif -6 < decimal_position <= 0:
        encoded = "0." + ("0" * -decimal_position) + digits
    else:
        mantissa = digits[0]
        if len(digits) > 1:
            mantissa += "." + digits[1:]
        scientific_exponent = decimal_position - 1
        sign = "+" if scientific_exponent >= 0 else ""
        encoded = f"{mantissa}e{sign}{scientific_exponent}"

    return "-" + encoded if negative else encoded
