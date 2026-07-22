"""Phase-A C1 model-activation core — shared types.

Path-safe refusal reasons suitable for later degraded/503 mapping.
Filesystem paths must never appear in messages, logs, or telemetry.

Public ActivationRefusal mapping must never preserve a path-bearing OSError in
``__context__`` / ``__cause__``: map inside ``except``, raise outside.
"""

from __future__ import annotations

import errno
import os
import stat
import threading
from enum import Enum
from typing import Callable, Final, Optional


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


# ---------------------------------------------------------------------------
# Freeze handle
# ---------------------------------------------------------------------------

# Member / freeze-leaf opens that may hit a FIFO must include O_NONBLOCK so
# fstat can reject SPECIAL_FILE without blocking for a writer.
_FREEZE_LEAF_FLAGS = (
    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
)
_FREEZE_DIR_FLAGS = (
    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
)


class FrozenBundle:
    """Service-private sealed freeze handle for a BUNDLE activation.

    Preferred handle is a held directory **fd** (not a public absolute path).
    There is **no** public mutable ``path`` attribute — a path string is
    re-openable / renamable / repointable and is not the immutability mechanism.

    The raw freeze dir fd is **not** exposed as a borrowed public attribute.
    Callers that need a directory fd must take a caller-owned duplicate via
    :meth:`dup_dir_fd` and close it themselves.

    Reads go through ``open_member`` / ``read_member`` relative to the freeze
    dir fd. ``cleanup()`` destroys owned freeze resources via the creation-time
    identity ledger (descriptor-relative; never ``shutil.rmtree(path)``).

    Concurrency: ``dup_dir_fd`` / ``open_member`` / ``cleanup`` share an internal
    lock and in-flight counter. Validation + ``os.dup`` of the bundle root run
    only while the root fd is protected from close, so concurrent cleanup cannot
    expose a recycled unrelated fd number to dup/open.
    """

    __slots__ = (
        "_dir_fd",
        "_lease",
        "_digest",
        "_file_count",
        "_aggregate_bytes",
        "_closed",
        "_lock",
        "_cond",
        "_in_flight",
        # True while a single serialized lease.release() is in progress.
        "_releasing",
        # Optional test hook: called after validate/in_flight++, before os.dup.
        "_after_validate_hook",
        # Test-only private residual name for path-redirect RED (not public API).
        "_backing_path",
    )

    def __init__(
        self,
        dir_fd: int,
        lease: object,
        digest: str,
        file_count: int,
        aggregate_bytes: int,
        *,
        backing_path: str = "",
    ) -> None:
        if dir_fd < 0:
            raise ValueError("dir_fd must be open")
        lock = threading.Lock()
        object.__setattr__(self, "_dir_fd", dir_fd)
        object.__setattr__(self, "_lease", lease)
        object.__setattr__(self, "_digest", digest)
        object.__setattr__(self, "_file_count", file_count)
        object.__setattr__(self, "_aggregate_bytes", aggregate_bytes)
        object.__setattr__(self, "_closed", False)
        object.__setattr__(self, "_lock", lock)
        object.__setattr__(self, "_cond", threading.Condition(lock))
        object.__setattr__(self, "_in_flight", 0)
        object.__setattr__(self, "_releasing", False)
        object.__setattr__(self, "_after_validate_hook", None)
        object.__setattr__(self, "_backing_path", backing_path)

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

    def _stable_root_dup(self) -> int:
        """Validate under lock, protect root from close, return caller-owned dup.

        ``cleanup`` waits for ``_in_flight == 0`` before closing the bundle root
        fd, so the number passed to ``os.dup`` cannot be recycled mid-call.
        """
        lock: threading.Lock = self._lock
        cond: threading.Condition = self._cond
        with lock:
            if self._closed or self._dir_fd < 0:
                raise ActivationRefusal(RefusalReason.FREEZE_FAILED, "closed")
            object.__setattr__(self, "_in_flight", int(self._in_flight) + 1)
            root = int(self._dir_fd)
        # in_flight > 0: cleanup will not close ``root`` until we release.
        try:
            hook: Optional[Callable[[], None]] = self._after_validate_hook
            if hook is not None:
                hook()
            # Cleanup may have set _closed and be waiting on in_flight; the root
            # number remains live until in_flight drains, so only re-check that
            # the bundle still owns this exact fd number (not yet closed/recycled).
            with lock:
                if self._dir_fd < 0 or int(self._dir_fd) != root:
                    raise ActivationRefusal(RefusalReason.FREEZE_FAILED, "closed")
            mapped: Optional[ActivationRefusal] = None
            dup = -1
            try:
                dup = os.dup(root)
            except OSError:
                mapped = ActivationRefusal(RefusalReason.FREEZE_FAILED, "dup failed")
            if mapped is not None:
                raise mapped
            return dup
        finally:
            with lock:
                object.__setattr__(
                    self, "_in_flight", max(0, int(self._in_flight) - 1)
                )
                cond.notify_all()

    def dup_dir_fd(self) -> int:
        """Return a caller-owned duplicate of the freeze root directory fd.

        The caller must ``os.close`` the returned fd. Concurrent ``cleanup()``
        closes only the bundle-owned fd after in-flight dups finish; a live
        duplicate remains valid until the caller closes it (inode stays alive).
        After cleanup, further ``dup_dir_fd`` calls refuse.
        """
        return self._stable_root_dup()

    def open_member(self, relpath: str) -> int:
        """Open a regular file under the freeze via the held dir fd.

        ``relpath`` is a relative POSIX path with no ``.`` / ``..`` / empty /
        absolute components (same raw domain as pins). Returns an owned fd.

        Acquires a stable root duplicate under the concurrency protocol so
        concurrent ``cleanup()`` cannot redirect opens to a recycled fd.
        """
        from src.core.model_activation.pin_domain import validate_raw_pin

        components = validate_raw_pin(relpath)
        base = self._stable_root_dup()
        owned: list[int] = []
        dir_fd = base
        try:
            for i, name in enumerate(components):
                is_last = i == len(components) - 1
                if not is_last:
                    mapped: Optional[ActivationRefusal] = None
                    next_fd = -1
                    try:
                        next_fd = os.open(name, _FREEZE_DIR_FLAGS, dir_fd=dir_fd)
                    except OSError:
                        mapped = ActivationRefusal(
                            RefusalReason.UNREADABLE, "freeze open"
                        )
                    if mapped is not None:
                        raise mapped
                    owned.append(next_fd)
                    dir_fd = next_fd
                    continue
                mapped_leaf: Optional[ActivationRefusal] = None
                leaf = -1
                try:
                    leaf = os.open(name, _FREEZE_LEAF_FLAGS, dir_fd=dir_fd)
                except OSError:
                    mapped_leaf = ActivationRefusal(
                        RefusalReason.UNREADABLE, "freeze open"
                    )
                if mapped_leaf is not None:
                    raise mapped_leaf
                mapped_st: Optional[ActivationRefusal] = None
                st: Optional[os.stat_result] = None
                try:
                    st = os.fstat(leaf)
                except OSError:
                    mapped_st = ActivationRefusal(
                        RefusalReason.UNREADABLE, "freeze fstat"
                    )
                if mapped_st is not None:
                    try:
                        os.close(leaf)
                    except OSError:
                        pass
                    raise mapped_st
                assert st is not None
                if not stat.S_ISREG(st.st_mode):
                    try:
                        os.close(leaf)
                    except OSError:
                        pass
                    raise ActivationRefusal(
                        RefusalReason.NOT_REGULAR_FILE, "freeze member"
                    )
                return leaf
            raise ActivationRefusal(RefusalReason.INTERNAL, "freeze open empty")
        finally:
            for fd in reversed(owned):
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                os.close(base)
            except OSError:
                pass

    def read_member(self, relpath: str) -> bytes:
        """Read a freeze member fully via the held dir fd (same-fd)."""
        from src.core.model_activation.digest import read_fd_bounded_once

        fd = self.open_member(relpath)
        try:
            mapped: Optional[ActivationRefusal] = None
            st: Optional[os.stat_result] = None
            try:
                st = os.fstat(fd)
            except OSError:
                mapped = ActivationRefusal(RefusalReason.UNREADABLE, "freeze fstat")
            if mapped is not None:
                raise mapped
            assert st is not None
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

    def cleanup(self) -> bool:
        """Close the freeze dir fd and destroy owned freeze resources.

        Waits for in-flight ``dup_dir_fd`` / ``open_member`` root dups, then
        closes the bundle root under the lock so the fd number cannot be
        recycled into a concurrent dup/open.

        The freeze lease is **retained until** ``lease.release()`` returns
        True. Incomplete release (``False``) leaves the lease installed so a
        later ``cleanup()`` retries the same object. Concurrent ``cleanup``
        callers serialize on a single in-flight release — none returns True
        while release is still running, and release is never invoked
        concurrently.

        Returns True only when cleanup completed (no owned residual requiring
        retry). Descriptor-relative scrub only; never ``shutil.rmtree(path)``.
        """
        lock: threading.Lock = self._lock
        cond: threading.Condition = self._cond
        lease: object = None
        with lock:
            # Serialize release attempts: wait out any in-flight release.
            while bool(self._releasing):
                cond.wait(timeout=0.05)

            # Fully done: closed and no lease left to retry.
            if self._closed and self._lease is None:
                return True

            if not self._closed:
                object.__setattr__(self, "_closed", True)
                # Drain in-flight stable dups before closing the *bundle* root
                # dup. The lease retains its original root fd and is responsible
                # for scrubbing/closing it in lease.release() — never detach
                # the lease root without close (that leaked the freeze tree).
                while int(self._in_flight) > 0:
                    cond.wait(timeout=0.05)
                fd = int(self._dir_fd)
                object.__setattr__(self, "_dir_fd", -1)
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass

            lease = self._lease
            if lease is None:
                return True
            # Claim exclusive release slot; lease stays installed until success.
            object.__setattr__(self, "_releasing", True)

        done = False
        try:
            if hasattr(lease, "release"):
                done = bool(lease.release())  # type: ignore[no-any-return]
            else:
                done = True
        finally:
            with lock:
                if done:
                    # Only drop the lease after proven successful release.
                    if self._lease is lease:
                        object.__setattr__(self, "_lease", None)
                object.__setattr__(self, "_releasing", False)
                cond.notify_all()
        return done

    def __del__(self) -> None:  # pragma: no cover - best-effort
        try:
            self.cleanup()
        except Exception:
            pass


def seal_freeze_tree_fd(dir_fd: int) -> None:
    """Make freeze tree owner-read-only via descriptor-relative fchmod.

    Files ``0400``, directories ``0500``. Walk is descriptor-relative from
    ``dir_fd`` (no path re-open of the freeze root).
    """
    if dir_fd < 0:
        raise ActivationRefusal(RefusalReason.FREEZE_FAILED, "seal bad fd")

    def _seal_dir(cur_fd: int) -> None:
        from src.core.model_activation.fd_dir import dirent_name_str, scandir_dir_fd

        it = scandir_dir_fd(cur_fd)
        try:
            for entry in it:
                name = dirent_name_str(entry.name)
                if name in (".", ".."):
                    continue
                mapped_open: Optional[ActivationRefusal] = None
                child = -1
                is_dir = False
                try:
                    child = os.open(name, _FREEZE_DIR_FLAGS, dir_fd=cur_fd)
                    is_dir = True
                except OSError:
                    try:
                        child = os.open(name, _FREEZE_LEAF_FLAGS, dir_fd=cur_fd)
                    except OSError:
                        mapped_open = ActivationRefusal(
                            RefusalReason.FREEZE_FAILED, "seal open"
                        )
                if mapped_open is not None:
                    raise mapped_open
                try:
                    mapped_st: Optional[ActivationRefusal] = None
                    st: Optional[os.stat_result] = None
                    try:
                        st = os.fstat(child)
                    except OSError:
                        mapped_st = ActivationRefusal(
                            RefusalReason.FREEZE_FAILED, "seal fstat"
                        )
                    if mapped_st is not None:
                        raise mapped_st
                    assert st is not None
                    if is_dir or stat.S_ISDIR(st.st_mode):
                        _seal_dir(child)
                        mapped_ch: Optional[ActivationRefusal] = None
                        try:
                            os.fchmod(child, 0o500)
                        except OSError as exc:
                            if exc.errno != errno.ENOENT:
                                mapped_ch = ActivationRefusal(
                                    RefusalReason.FREEZE_FAILED, "seal dir"
                                )
                        if mapped_ch is not None:
                            raise mapped_ch
                    elif stat.S_ISREG(st.st_mode):
                        mapped_cf: Optional[ActivationRefusal] = None
                        try:
                            os.fchmod(child, 0o400)
                        except OSError as exc:
                            if exc.errno != errno.ENOENT:
                                mapped_cf = ActivationRefusal(
                                    RefusalReason.FREEZE_FAILED, "seal file"
                                )
                        if mapped_cf is not None:
                            raise mapped_cf
                    else:
                        raise ActivationRefusal(
                            RefusalReason.SPECIAL_FILE, "seal non-regular"
                        )
                finally:
                    try:
                        os.close(child)
                    except OSError:
                        pass
        finally:
            try:
                it.close()
            except Exception:
                pass

        mapped_root: Optional[ActivationRefusal] = None
        try:
            os.fchmod(cur_fd, 0o500)
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                mapped_root = ActivationRefusal(
                    RefusalReason.FREEZE_FAILED, "seal root"
                )
        if mapped_root is not None:
            raise mapped_root

    _seal_dir(dir_fd)


__all__ = [
    "ActivationRefusal",
    "ArtifactKind",
    "BoundPolicy",
    "FrozenBundle",
    "PinRecord",
    "RefusalReason",
    "TerminalKind",
    "seal_freeze_tree_fd",
]
