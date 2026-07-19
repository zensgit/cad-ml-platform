"""Raw POSIX pin-string domain validation.

MUST run on the raw pin BEFORE any Path/PurePosixPath/os.path normalization
and BEFORE either openat2 or the component-walk resolver. Both resolvers share
this pre-gate so they reject the identical raw input set.
"""

from __future__ import annotations

from typing import Tuple

from src.core.model_activation.types import ActivationRefusal, RefusalReason


def validate_raw_pin(raw_pin: str) -> Tuple[str, ...]:
    """Validate raw pin and return non-empty relative components.

    Rejects:
    - non-str / empty
    - NUL (0x00) anywhere
    - absolute (leading ``/``)
    - empty components (``//``, trailing ``/``, leading ``/``)
    - ``.`` or ``..`` components

    Deliberately does **not** call Path/os.path normalizers — those collapse
    ``//`` and ``.`` and would hide the illegal components this check exists
    to catch.
    """
    if not isinstance(raw_pin, str):
        raise ActivationRefusal(
            RefusalReason.RAW_PIN_INVALID, "pin must be a str"
        )
    if raw_pin == "":
        raise ActivationRefusal(RefusalReason.RAW_PIN_INVALID, "empty pin")
    if "\x00" in raw_pin:
        raise ActivationRefusal(RefusalReason.RAW_PIN_INVALID, "nul in pin")
    if raw_pin.startswith("/"):
        # Absolute prefix; leading '/' also yields an empty first component.
        raise ActivationRefusal(RefusalReason.RAW_PIN_INVALID, "absolute pin")

    parts = raw_pin.split("/")
    for part in parts:
        if part == "":
            raise ActivationRefusal(
                RefusalReason.RAW_PIN_INVALID, "empty component"
            )
        if part == "." or part == "..":
            raise ActivationRefusal(
                RefusalReason.RAW_PIN_INVALID, "dot component"
            )
    return tuple(parts)


def validate_readdir_name(name: str) -> None:
    """Reject illegal readdir entry names during bundle traversal.

    Never descend into ``.`` / ``..``. Reject empty names and names containing
    ``/`` or NUL (a readdir name must be a single path component).
    """
    if not isinstance(name, str):
        raise ActivationRefusal(
            RefusalReason.MALFORMED_ENTRY, "entry name not str"
        )
    if name == "" or name == "." or name == "..":
        raise ActivationRefusal(
            RefusalReason.MALFORMED_ENTRY, "illegal entry name"
        )
    if "/" in name or "\x00" in name:
        raise ActivationRefusal(
            RefusalReason.MALFORMED_ENTRY, "illegal entry name"
        )


__all__ = ["validate_raw_pin", "validate_readdir_name"]
