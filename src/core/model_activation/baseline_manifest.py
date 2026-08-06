"""Phase-A C3 baseline pin-manifest loader.

Reads the ratified baseline pin manifest that locks the full
``(logical_activation_id, artifact_id, kind, digest, store_relpath)`` tuple for
every artifact a deployment is allowed to activate. The manifest is **not**
runtime-repointable; real production digests are supplied at ENABLEMENT (owner
gate, out of Phase-A scope). This module ships the MECHANISM plus the default
NO-PIN posture, and a ``source`` override so tests/fixtures can inject a known
pin set.

Fail-closed contract
--------------------
* **Env/arg unset, or the manifest file is absent** → return ``()`` (empty).
  Empty is the default production NO-PIN posture: the controlled store has zero
  pins and every activation degrades — it NEVER falls back to an unverified raw
  load.
* **Manifest present but unreadable / malformed** → raise :class:`ValueError`.
  An unreadable or corrupt manifest must NEVER masquerade as "no pins
  configured" (which would silently degrade every family and mask the
  corruption). It fails LOUD instead.

JSON schema
-----------
Top-level: a JSON **list** of pin objects. Each object has EXACTLY these
string fields (unknown fields are rejected fail-closed)::

    {
      "logical_activation_id": "<non-empty logical id>",
      "artifact_id":           "<non-empty artifact id>",
      "kind":                  "single_file" | "bundle",
      "digest":                "<64-char lowercase hex sha-256>",
      "store_relpath":         "<relative POSIX pin under the store root>"
    }

``(logical_activation_id, artifact_id)`` must be unique across the manifest;
a duplicate key is rejected fail-closed. ``digest`` / ``kind`` are validated by
:class:`~src.core.model_activation.types.PinRecord` itself (let it raise), and
``store_relpath`` is validated against the raw-pin domain here (defense in depth
for attacker-influenceable config) — both surface as :class:`ValueError`.

Path safety
-----------
This is L3 code: no filesystem path (the manifest path, a ``store_relpath``, or
any resolved path) ever appears in an exception message or log. Errors identify
the offending entry by its list index only.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Set, Tuple

from src.core.model_activation.pin_domain import validate_raw_pin
from src.core.model_activation.types import (
    ActivationRefusal,
    PinRecord,
)

ENV_BASELINE_MANIFEST = "MODEL_ACTIVATION_BASELINE_MANIFEST"

_REQUIRED_FIELDS: Tuple[str, ...] = (
    "logical_activation_id",
    "artifact_id",
    "kind",
    "digest",
    "store_relpath",
)
_REQUIRED_FIELD_SET = frozenset(_REQUIRED_FIELDS)


def load_baseline_pins(source: Optional[str] = None) -> Tuple[PinRecord, ...]:
    """Load the baseline pin set from a JSON manifest.

    ``source`` (if given) or the ``MODEL_ACTIVATION_BASELINE_MANIFEST`` env var
    names the manifest path. Returns ``()`` when unset or the file is absent
    (default NO-PIN posture). Raises :class:`ValueError` when the manifest is
    present but unreadable or malformed (fail LOUD — never silent-empty).
    """
    manifest_path = source if source is not None else os.environ.get(ENV_BASELINE_MANIFEST)
    if not manifest_path:
        return ()

    # Race-free presence check: a genuinely absent file → empty (degraded);
    # anything else present-but-unopenable (incl. a directory) → fail loud.
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except FileNotFoundError:
        return ()
    except OSError:
        raise ValueError("baseline manifest is present but could not be read") from None

    try:
        parsed = json.loads(text)
    except ValueError:
        # json.JSONDecodeError is a ValueError subclass; re-raise path-free.
        raise ValueError("baseline manifest is not valid JSON") from None

    if not isinstance(parsed, list):
        raise ValueError("baseline manifest must be a JSON list of pin objects")

    pins: List[PinRecord] = []
    seen: Set[Tuple[str, str]] = set()
    for index, entry in enumerate(parsed):
        pin = _entry_to_pin(entry, index)
        key = (pin.logical_activation_id, pin.artifact_id)
        if key in seen:
            raise ValueError(
                f"baseline manifest entry at index {index} duplicates an earlier "
                "(logical_activation_id, artifact_id) key; keys must be unique"
            )
        seen.add(key)
        pins.append(pin)
    return tuple(pins)


def _entry_to_pin(entry: object, index: int) -> PinRecord:
    """Validate a single manifest entry into a :class:`PinRecord` (fail-closed)."""
    if not isinstance(entry, dict):
        raise ValueError(f"baseline manifest entry at index {index} must be a JSON object")

    keys = set(entry.keys())
    missing = _REQUIRED_FIELD_SET - keys
    if missing:
        raise ValueError(
            f"baseline manifest entry at index {index} is missing required field(s): "
            + ", ".join(sorted(missing))
        )
    unknown = keys - _REQUIRED_FIELD_SET
    if unknown:
        raise ValueError(
            f"baseline manifest entry at index {index} has unknown field(s): "
            + ", ".join(sorted(unknown))
        )
    for field in _REQUIRED_FIELDS:
        if not isinstance(entry[field], str):
            raise ValueError(
                f"baseline manifest entry at index {index} field {field!r} must be a string"
            )

    # Defense in depth: reject a traversal / absolute / malformed store_relpath
    # in attacker-influenceable config here, so it surfaces as a loud ValueError
    # at load time (not only later as an ActivationRefusal at store construction).
    try:
        validate_raw_pin(entry["store_relpath"])
    except ActivationRefusal:
        raise ValueError(
            f"baseline manifest entry at index {index} has an invalid store_relpath"
        ) from None

    # PinRecord validates digest (64-hex) and kind (single_file|bundle) itself;
    # let its ValueError propagate (path-free), tagged with the entry index.
    try:
        return PinRecord(
            logical_activation_id=entry["logical_activation_id"],
            artifact_id=entry["artifact_id"],
            kind=entry["kind"],
            digest=entry["digest"],
            store_relpath=entry["store_relpath"],
        )
    except ValueError as exc:
        raise ValueError(
            f"baseline manifest entry at index {index} is invalid: {exc}"
        ) from None


__all__ = ["ENV_BASELINE_MANIFEST", "load_baseline_pins"]
