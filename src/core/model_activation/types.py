"""Phase-A C1 model-activation core — shared types.

Path-safe refusal reasons suitable for later degraded/503 mapping.
Filesystem paths must never appear in messages, logs, or telemetry.
"""

from __future__ import annotations

import errno
import os
import shutil
import stat
from enum import Enum
from typing import Final, Optional


class ArtifactKind(str, Enum):
    """Pinned artifact KIND. Narrow and impossible to confuse."""

    SINGLE_FILE = "single_file"
    BUNDLE = "bundle"


class TerminalKind(str, Enum):
    """What the resolver's terminal node must be."""

    REGULAR_FILE = "regular_file"
    DIRECTORY = "directory"


class RefusalReason(str, Enum):
    """Typed activation refusal reasons (no filesystem paths)."""

    PIN_ABSENT = "pin_absent"
    PIN_UNKNOWN = "pin_unknown"
    KIND_MISMATCH = "kind_mismatch"
    DIGEST_MISMATCH = "digest_mismatch"
    DIGEST_INVALID = "digest_invalid"
    RAW_PIN_INVALID = "raw_pin_invalid"
    CONTAINMENT = "containment"
    ARTIFACT_MISSING = "artifact_missing"
    NOT_REGULAR_FILE = "not_regular_file"
    NOT_DIRECTORY = "not_directory"
    SYMLINK_REJECTED = "symlink_rejected"
    SPECIAL_FILE = "special_file"
    OVERSIZE = "oversize"
    GROWING_FILE = "growing_file"
    BUNDLE_FILE_COUNT = "bundle_file_count"
    BUNDLE_PER_FILE_BYTES = "bundle_per_file_bytes"
    BUNDLE_AGGREGATE_BYTES = "bundle_aggregate_bytes"
    BUNDLE_TOTAL_DIRENTS = "bundle_total_dirents"
    BUNDLE_DIRECTORY_COUNT = "bundle_directory_count"
    BUNDLE_DEPTH = "bundle_depth"
    BUNDLE_RELPATH_BYTES = "bundle_relpath_bytes"
    NON_UTF8_ENTRY = "non_utf8_entry"
    MALFORMED_ENTRY = "malformed_entry"
    UNREADABLE = "unreadable"
    FREEZE_FAILED = "freeze_failed"
    FREEZE_MUTATED = "freeze_mutated"
    INTERNAL = "internal"


class ActivationRefusal(Exception):
    """Fail-closed activation refusal. ``reason`` is path-safe for mapping."""

    def __init__(self, reason: RefusalReason, detail: str = "") -> None:
        self.reason: Final[RefusalReason] = reason
        # detail must never contain filesystem paths
        self.detail: Final[str] = detail
        super().__init__(f"{reason.value}" + (f": {detail}" if detail else ""))


class PinRecord:
    """Immutable pin record. No filesystem absolute path fields.

    ``store_relpath`` is the RAW POSIX relative pin under the controlled store
    root (validated before any Path/os.path normalization).
    """

    __slots__ = (
        "logical_activation_id",
        "artifact_id",
        "kind",
        "digest",
        "store_relpath",
    )

    def __init__(
        self,
        logical_activation_id: str,
        artifact_id: str,
        kind: ArtifactKind,
        digest: str,
        store_relpath: str,
    ) -> None:
        if not logical_activation_id or not artifact_id:
            raise ValueError("logical_activation_id and artifact_id must be non-empty")
        dig = digest.lower()
        if len(dig) != 64 or any(c not in "0123456789abcdef" for c in dig):
            raise ValueError("digest must be 64-char lowercase hex SHA-256")
        if not isinstance(kind, ArtifactKind):
            kind = ArtifactKind(kind)
        object.__setattr__(self, "logical_activation_id", logical_activation_id)
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "digest", dig)
        object.__setattr__(self, "store_relpath", store_relpath)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("PinRecord is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("PinRecord is immutable")


class BoundPolicy:
    """Server-owned resource bounds (not caller/env input)."""

    __slots__ = (
        "max_single_file_bytes",
        "max_bundle_file_count",
        "max_bundle_per_file_bytes",
        "max_bundle_aggregate_bytes",
        "max_total_dirents",
        "max_directories",
        "max_depth",
        "max_relpath_utf8_bytes",
    )

    def __init__(
        self,
        max_single_file_bytes: int = 512 * 1024 * 1024,
        max_bundle_file_count: int = 10_000,
        max_bundle_per_file_bytes: int = 256 * 1024 * 1024,
        max_bundle_aggregate_bytes: int = 2 * 1024 * 1024 * 1024,
        max_total_dirents: int = 50_000,
        max_directories: int = 10_000,
        max_depth: int = 64,
        max_relpath_utf8_bytes: int = 4096,
    ) -> None:
        vals = {
            "max_single_file_bytes": max_single_file_bytes,
            "max_bundle_file_count": max_bundle_file_count,
            "max_bundle_per_file_bytes": max_bundle_per_file_bytes,
            "max_bundle_aggregate_bytes": max_bundle_aggregate_bytes,
            "max_total_dirents": max_total_dirents,
            "max_directories": max_directories,
            "max_depth": max_depth,
            "max_relpath_utf8_bytes": max_relpath_utf8_bytes,
        }
        for name, val in vals.items():
            if not isinstance(val, int) or val < 1:
                raise ValueError(f"{name} must be a positive int")
            object.__setattr__(self, name, val)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("BoundPolicy is immutable")


class FrozenBundle:
    """Service-private sealed freeze handle for a BUNDLE activation.

    Preferred handle is a held directory **fd** (not a public absolute path).
    There is **no** public mutable ``path`` attribute — a path string is
    re-openable / renamable / repointable and is not the immutability mechanism.

    Reads go through ``open_member`` / ``read_member`` relative to the freeze
    dir fd. ``cleanup()`` closes the fd and destroys the private backing tree.
    """

    __slots__ = (
        "_dir_fd",
        "_backing_path",
        "_digest",
        "_file_count",
        "_aggregate_bytes",
        "_closed",
    )

    def __init__(
        self,
        dir_fd: int,
        backing_path: str,
        digest: str,
        file_count: int,
        aggregate_bytes: int,
    ) -> None:
        if dir_fd < 0:
            raise ValueError("dir_fd must be open")
        object.__setattr__(self, "_dir_fd", dir_fd)
        object.__setattr__(self, "_backing_path", backing_path)
        object.__setattr__(self, "_digest", digest)
        object.__setattr__(self, "_file_count", file_count)
        object.__setattr__(self, "_aggregate_bytes", aggregate_bytes)
        object.__setattr__(self, "_closed", False)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("FrozenBundle fields are immutable")

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def file_count(self) -> int:
        return self._file_count

    @property
    def aggregate_bytes(self) -> int:
        return self._aggregate_bytes

    @property
    def dir_fd(self) -> int:
        """Borrowed freeze root directory fd. Do not close; use cleanup()."""
        if self._closed or self._dir_fd < 0:
            raise ActivationRefusal(RefusalReason.FREEZE_FAILED, "closed")
        return self._dir_fd

    def open_member(self, relpath: str) -> int:
        """Open a regular file under the freeze via the held dir fd.

        ``relpath`` is a relative POSIX path with no ``.`` / ``..`` / empty /
        absolute components (same raw domain as pins). Returns an owned fd.
        """
        from src.core.model_activation.pin_domain import validate_raw_pin

        if self._closed or self._dir_fd < 0:
            raise ActivationRefusal(RefusalReason.FREEZE_FAILED, "closed")
        components = validate_raw_pin(relpath)
        dir_fd = self._dir_fd
        owned: list[int] = []
        try:
            for i, name in enumerate(components):
                is_last = i == len(components) - 1
                if not is_last:
                    try:
                        next_fd = os.open(
                            name,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=dir_fd,
                        )
                    except OSError as exc:
                        raise ActivationRefusal(
                            RefusalReason.UNREADABLE, "freeze open"
                        ) from exc
                    owned.append(next_fd)
                    dir_fd = next_fd
                    continue
                try:
                    leaf = os.open(
                        name,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=dir_fd,
                    )
                except OSError as exc:
                    raise ActivationRefusal(
                        RefusalReason.UNREADABLE, "freeze open"
                    ) from exc
                try:
                    st = os.fstat(leaf)
                except OSError as exc:
                    os.close(leaf)
                    raise ActivationRefusal(
                        RefusalReason.UNREADABLE, "freeze fstat"
                    ) from exc
                if not stat.S_ISREG(st.st_mode):
                    os.close(leaf)
                    raise ActivationRefusal(
                        RefusalReason.NOT_REGULAR_FILE, "freeze member"
                    )
                return leaf
        finally:
            for fd in reversed(owned):
                try:
                    os.close(fd)
                except OSError:
                    pass
        raise ActivationRefusal(RefusalReason.INTERNAL, "freeze open empty")

    def read_member(self, relpath: str) -> bytes:
        """Read a freeze member fully via the held dir fd (same-fd)."""
        from src.core.model_activation.digest import read_fd_bounded_once

        fd = self.open_member(relpath)
        try:
            st = os.fstat(fd)
            # Freeze already size-bounded; allow empty files (expect_size=0).
            return read_fd_bounded_once(
                fd,
                max(int(st.st_size), 1),
                expect_size=int(st.st_size),
            )
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def cleanup(self) -> None:
        """Close the freeze dir fd and destroy the private backing tree."""
        if self._closed:
            return
        object.__setattr__(self, "_closed", True)
        fd = self._dir_fd
        object.__setattr__(self, "_dir_fd", -1)
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        path = self._backing_path
        if path:
            # May need write bits to rmtree a sealed tree.
            make_tree_deletable(path)
            shutil.rmtree(path, ignore_errors=True)

    def __del__(self) -> None:  # pragma: no cover - best-effort
        try:
            self.cleanup()
        except Exception:
            pass


def make_tree_deletable(root: str) -> None:
    """Best-effort restore write bits so rmtree can unlink a sealed freeze."""
    try:
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            for name in filenames:
                p = os.path.join(dirpath, name)
                try:
                    os.chmod(p, 0o600)
                except OSError:
                    pass
            for name in dirnames:
                p = os.path.join(dirpath, name)
                try:
                    os.chmod(p, 0o700)
                except OSError:
                    pass
        try:
            os.chmod(root, 0o700)
        except OSError:
            pass
    except OSError:
        pass


def seal_freeze_tree(root: str) -> None:
    """Make freeze tree owner-read-only (files 0400, dirs 0500)."""
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            p = os.path.join(dirpath, name)
            try:
                os.chmod(p, 0o400)
            except OSError as exc:
                if exc.errno != errno.ENOENT:
                    raise ActivationRefusal(
                        RefusalReason.FREEZE_FAILED, "seal file"
                    ) from exc
        for name in dirnames:
            p = os.path.join(dirpath, name)
            try:
                os.chmod(p, 0o500)
            except OSError as exc:
                if exc.errno != errno.ENOENT:
                    raise ActivationRefusal(
                        RefusalReason.FREEZE_FAILED, "seal dir"
                    ) from exc
    try:
        os.chmod(root, 0o500)
    except OSError as exc:
        raise ActivationRefusal(RefusalReason.FREEZE_FAILED, "seal root") from exc


__all__ = [
    "ActivationRefusal",
    "ArtifactKind",
    "BoundPolicy",
    "FrozenBundle",
    "PinRecord",
    "RefusalReason",
    "TerminalKind",
    "make_tree_deletable",
    "seal_freeze_tree",
]
