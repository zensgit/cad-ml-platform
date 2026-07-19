"""SHA-256 helpers and the ratified tree-digest-v1 encoding."""

from __future__ import annotations

import hashlib
import os
from typing import Iterable, List, Sequence, Tuple

from src.core.model_activation.types import ActivationRefusal, RefusalReason

_UNIT_SEP = b"\x1f"
_REC_SEP = b"\x00"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_fd_bounded_once(fd: int, max_bytes: int, *, expect_size: int | None = None) -> bytes:
    """Read once from an already-open fd with a hard byte bound.

    If ``expect_size`` is set (from a prior fstat on this same fd), the read
    must yield exactly that many bytes and no more — growing/truncating during
    the read is refused (``GROWING_FILE`` / size instability).
    """
    if max_bytes < 1:
        raise ActivationRefusal(RefusalReason.OVERSIZE, "max_bytes < 1")
    if expect_size is not None:
        if expect_size < 0:
            raise ActivationRefusal(RefusalReason.UNREADABLE, "negative size")
        if expect_size > max_bytes:
            raise ActivationRefusal(RefusalReason.OVERSIZE, "fstat oversize")

    chunks: List[bytes] = []
    total = 0
    target = expect_size if expect_size is not None else max_bytes
    # Read in chunks; never exceed max_bytes.
    while total < target:
        to_read = min(1024 * 1024, target - total)
        try:
            chunk = os.read(fd, to_read)
        except OSError as exc:
            raise ActivationRefusal(RefusalReason.UNREADABLE, "read failed") from exc
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ActivationRefusal(RefusalReason.OVERSIZE, "read oversize")
        chunks.append(chunk)

    data = b"".join(chunks)

    if expect_size is not None:
        if len(data) != expect_size:
            raise ActivationRefusal(
                RefusalReason.GROWING_FILE, "size changed during read"
            )
        # One more probe: if the file grew past fstat size, refuse.
        try:
            extra = os.read(fd, 1)
        except OSError as exc:
            raise ActivationRefusal(RefusalReason.UNREADABLE, "read probe") from exc
        if extra:
            raise ActivationRefusal(RefusalReason.GROWING_FILE, "grew past fstat")
    else:
        if len(data) > max_bytes:
            raise ActivationRefusal(RefusalReason.OVERSIZE, "read oversize")

    return data


def tree_digest_v1_records(entries: Sequence[Tuple[str, str]]) -> bytes:
    """Build the canonical tree-digest-v1 payload.

    ``entries`` is a sequence of ``(posix_relpath, lowercase_hex_sha256)``.
    Only regular-file members participate (caller filters). Records are sorted
    bytewise by UTF-8 relpath and joined by ``0x00``.

    Each record::
        ascii_decimal(len(utf8_relpath)) · 0x1F · relpath_bytes · 0x1F · hex_digest
    """
    records: List[Tuple[bytes, bytes]] = []
    for relpath, hex_digest in entries:
        if not isinstance(relpath, str) or not isinstance(hex_digest, str):
            raise ActivationRefusal(RefusalReason.INTERNAL, "bad digest entry")
        try:
            rel_b = relpath.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ActivationRefusal(
                RefusalReason.NON_UTF8_ENTRY, "relpath not utf-8"
            ) from exc
        dig = hex_digest.lower()
        if len(dig) != 64 or any(c not in "0123456789abcdef" for c in dig):
            raise ActivationRefusal(RefusalReason.DIGEST_INVALID, "entry digest")
        rec = (
            str(len(rel_b)).encode("ascii")
            + _UNIT_SEP
            + rel_b
            + _UNIT_SEP
            + dig.encode("ascii")
        )
        records.append((rel_b, rec))
    records.sort(key=lambda item: item[0])
    return _REC_SEP.join(rec for _, rec in records)


def tree_digest_v1(entries: Sequence[Tuple[str, str]]) -> str:
    """SHA-256 hex of the tree-digest-v1 canonical encoding."""
    return sha256_hex(tree_digest_v1_records(entries))


def tree_digest_v1_from_file_bytes(
    files: Iterable[Tuple[str, bytes]],
) -> str:
    """Convenience: compute tree-digest-v1 from ``(relpath, file_bytes)``."""
    entries = [(rel, sha256_hex(data)) for rel, data in files]
    return tree_digest_v1(entries)


__all__ = [
    "read_fd_bounded_once",
    "sha256_hex",
    "tree_digest_v1",
    "tree_digest_v1_from_file_bytes",
    "tree_digest_v1_records",
]
