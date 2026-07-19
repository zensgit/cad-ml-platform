"""Directory listing via an already-open directory fd.

Python 3.10+ Unix ``os.listdir(dir_fd)`` — no ctypes, no ``/proc`` path hop.
Any OSError is fail-closed; never return a silent partial enumeration.
"""

from __future__ import annotations

import os
from typing import List

from src.core.model_activation.types import ActivationRefusal, RefusalReason


def list_dir_fd(dir_fd: int) -> List[str]:
    """Return all entry names for ``dir_fd`` (including ``.`` / ``..`` if present).

    Callers open members with ``openat`` on the original ``dir_fd``.
    """
    if not isinstance(dir_fd, int) or dir_fd < 0:
        raise ActivationRefusal(RefusalReason.INTERNAL, "bad dir_fd")
    try:
        # Python 3.10+ Unix: path may be an int dir_fd.
        names = os.listdir(dir_fd)
    except OSError as exc:
        raise ActivationRefusal(RefusalReason.UNREADABLE, "readdir") from exc
    except TypeError as exc:
        # Non-int / unsupported platform signature.
        raise ActivationRefusal(RefusalReason.UNREADABLE, "readdir") from exc

    if not isinstance(names, list):
        raise ActivationRefusal(RefusalReason.UNREADABLE, "readdir type")

    out: List[str] = []
    for name in names:
        if isinstance(name, bytes):
            try:
                name = name.decode("utf-8")
            except UnicodeDecodeError as uexc:
                raise ActivationRefusal(
                    RefusalReason.NON_UTF8_ENTRY, "entry name"
                ) from uexc
        if not isinstance(name, str):
            raise ActivationRefusal(RefusalReason.MALFORMED_ENTRY, "name type")
        if "\x00" in name:
            raise ActivationRefusal(RefusalReason.MALFORMED_ENTRY, "nul name")
        out.append(name)
    return out


__all__ = ["list_dir_fd"]
